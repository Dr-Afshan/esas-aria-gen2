"""
ESAS Live Demo

Fallback order:
  1. Aria Gen 2 glasses (requires SDK live streaming — not supported on Mac,
     RuntimeError: (2) Not implemented). Falls back automatically.
  2. Laptop microphone (mono, no direction sensing)
  3. Demo mode (press 1-5 to trigger manual alerts)

Usage:
  python esas_live.py              # auto-detect
  python esas_live.py --demo       # demo mode only (no mic needed)
  python esas_live.py --vrs FILE   # replay a VRS recording

Keys: 1-5 = trigger alert   ESC = quit   F = toggle fullscreen

Note: Live streaming from Aria Gen 2 is not supported on macOS with
SDK 2.2.0. Use CLI recording (aria recording start) and replay with
--vrs flag instead. Direction sensing requires Aria Gen 2 VRS files.
"""
import pygame, sys, time, threading, queue, argparse
import numpy as np
from pathlib import Path
sys.path.append('.')

# ── Args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--room',       default='')
parser.add_argument('--fullscreen', action='store_true')
parser.add_argument('--demo',       action='store_true')
parser.add_argument('--vrs',        default='',
    help='Path to VRS file for playback demo')
args = parser.parse_args()

# ── Shared state (lists so threads can mutate) ────────────────────
alert_q  = queue.Queue()
mode_txt = ['Starting...']
mic_lv   = [0.0]
history  = []
current  = [None]
show_t   = [0.0]

PRIORITY_COLOR = {
    'HIGH':   (210, 40,  40),
    'MEDIUM': (200, 120, 20),
    'LOW':    (40,  160, 60),
}
MANUAL = {
    '1': ('FIRE ALARM',  'HIGH',   'LEFT',  'Kitchen', 0.94),
    '2': ('PHONE RING',  'MEDIUM', 'FRONT', 'Office',  0.87),
    '3': ('CAR HORN',    'HIGH',   'RIGHT', 'Street',  0.91),
    '4': ('BABY CRYING', 'MEDIUM', 'LEFT',  'Bedroom', 0.82),
    '5': ('SIREN',       'HIGH',   'FRONT', 'Outdoor', 0.95),
}
ARROWS = {
    'FAR LEFT': '<<  FAR LEFT',
    'LEFT':     '<   LEFT',
    'FRONT':    '^   FRONT',
    'RIGHT':    'RIGHT   >',
    'FAR RIGHT':'FAR RIGHT  >>',
    'UNKNOWN':  'NEARBY',  # mono mic — no direction possible
}

def push(sound, priority, direction, room, conf, source):
    col = PRIORITY_COLOR.get(priority, (90,90,110))
    alert_q.put((sound, priority, direction,
                 room, conf, col, source))
    print(f"ALERT [{source}] [{priority}] {sound} "
          f"— {direction} {int(conf*100)}%")

# ── Audio thread ──────────────────────────────────────────────────
def audio_thread():
    # Step 1: load engine (slow — do in background)
    engine = None
    try:
        from sound_alert_system import AlertEngine
        engine = AlertEngine()
        if args.room:
            engine.set_room_context(args.room if args.room else 'unknown')
        print("ESAS engine loaded.")
    except Exception as e:
        print(f"ESAS engine not available: {e}")
        mode_txt[0] = 'DEMO MODE — ESAS engine missing'
        return

    # Allowlist — what PANNs labels map to our hazard sounds
    ALLOWED = {
        # HIGH priority sounds
        'fire alarm', 'smoke detector', 'smoke alarm', 'siren',
        'emergency vehicle', 'police siren', 'ambulance',
        'alarm', 'fire', 'screaming', 'explosion', 'gunshot',
        'glass break', 'glass shatter', 'car alarm',
        # MEDIUM priority sounds
        'phone', 'telephone', 'ringtone', 'doorbell',
        'door knock', 'knock', 'baby', 'crying', 'infant',
        'dog', 'bark', 'cat', 'clock alarm', 'alarm clock',
        'vacuum', 'car horn', 'horn', 'honking',
        # PANNs sometimes confuses these — include them
        'bell', 'ringing', 'ring', 'buzz', 'beep',
        'whimper', 'wail', 'whine', 'squeal',
        'vehicle horn', 'toot',
    }

    # Remap PANNs labels → correct display name + priority
    LABEL_REMAP = {
        # Fire alarm
        'fire alarm':             ('FIRE ALARM',  'HIGH'),
        'smoke detector':         ('FIRE ALARM',  'HIGH'),
        'smoke alarm':            ('FIRE ALARM',  'HIGH'),
        'alarm':                  ('FIRE ALARM',  'HIGH'),
        # Siren
        'siren':                  ('SIREN',        'HIGH'),
        'emergency vehicle':      ('SIREN',        'HIGH'),
        'police siren':           ('SIREN',        'HIGH'),
        # Car horn — only specific labels
        'car horn':               ('CAR HORN',    'HIGH'),
        'vehicle horn':           ('CAR HORN',    'HIGH'),
        'honking':                ('CAR HORN',    'HIGH'),
        'horn':                   ('CAR HORN',    'HIGH'),
        'toot':                   ('CAR HORN',    'HIGH'),
        # Phone ring — bells, rings, clocks (PANNs confuses these with phone)
        'telephone':              ('PHONE RING',  'MEDIUM'),
        'telephone bell ringing': ('PHONE RING',  'MEDIUM'),
        'ringtone':               ('PHONE RING',  'MEDIUM'),
        'ringing':                ('PHONE RING',  'MEDIUM'),
        'bell':                   ('PHONE RING',  'MEDIUM'),
        'bicycle bell':           ('PHONE RING',  'MEDIUM'),
        'clock alarm':            ('PHONE RING',  'MEDIUM'),
        'alarm clock':            ('PHONE RING',  'MEDIUM'),
        'doorbell':               ('PHONE RING',  'MEDIUM'),
        # Baby crying
        'baby cry':               ('BABY CRYING', 'MEDIUM'),
        'infant cry':             ('BABY CRYING', 'MEDIUM'),
        'crying':                 ('BABY CRYING', 'MEDIUM'),
        'cat':                    ('BABY CRYING', 'MEDIUM'),
        'whimper':                ('BABY CRYING', 'MEDIUM'),
        'wail':                   ('BABY CRYING', 'MEDIUM'),
        'squeal':                 ('BABY CRYING', 'MEDIUM'),
        # Dog
        'dog':                    ('DOG BARK',    'MEDIUM'),
        'bark':                   ('DOG BARK',    'MEDIUM'),
    }

    # Hard filter — never show these regardless of confidence
    NEVER_SHOW = {
        'vehicle', 'music', 'speech', 'animal', 'squawk',
        'fowl', 'mouse', 'rodent', 'bird', 'train',
        'throat', 'cough', 'sneeze', 'breathing',
        'sine wave', 'scissors', 'chink', 'dishes',
        'bleat', 'goose', 'turkey', 'owl', 'pig',
        'engine', 'bus', 'truck', 'bicycle',
        'gargling', 'burping', 'hiccup', 'yawn',
    }

    def run(audio, start_t, src):
        if np.max(np.abs(audio)) < 0.005:
            return
        try:
            ts    = (time.time()-start_t)*1000
            alert = engine.process_window(
                audio.reshape(1,-1), timestamp_ms=ts)
            if alert and alert.events:
                top = alert.events[0]
                label_lower = top.label.lower()
                # Only show if label matches allowed list
                is_allowed = any(a in label_lower for a in ALLOWED)
                # Also allow HIGH priority sounds above 50%
                is_high_conf = (top.priority == 'HIGH'
                                and top.confidence >= 0.50)
                label_lower = top.label.lower()

                # Hard filter — never show background noise
                if any(n in label_lower for n in NEVER_SHOW):
                    print(f"  Filtered: [{top.priority}] "
                          f"{top.label} {int(top.confidence*100)}%")
                    return

                # Check if label is in remap or allowlist
                in_remap   = label_lower in LABEL_REMAP
                in_allowed = any(a in label_lower for a in ALLOWED)

                if (in_remap or in_allowed or is_high_conf)                         and top.confidence >= 0.25:
                    if in_remap:
                        display_label, display_priority =                             LABEL_REMAP[label_lower]
                    else:
                        display_label    = top.label.upper()
                        display_priority = top.priority
                    # Use ESAS room context if available, else show source
                    room_display = alert.room_context                         if alert.room_context and alert.room_context != 'unknown'                         else ('Glasses' if src == 'GLASSES' else 'Laptop Mic')
                    push(display_label, display_priority,
                         alert.direction_label,
                         room_display,
                         top.confidence, src)
                else:
                    if top.confidence >= 0.25:
                        print(f"  Filtered: [{top.priority}] "
                              f"{top.label} {int(top.confidence*100)}%")
        except Exception as e:
            print(f"Detection error: {e}")

    # Step 2: try Aria Gen 2 using Gen 2 SDK
    glasses_ok = False
    try:
        import aria.sdk_gen2 as sdk_gen2
        import aria.stream_receiver as receiver
        from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord

        mode_txt[0] = 'Connecting to Aria Gen 2...'
        print("Trying Aria Gen 2 glasses...")

        # Connect to device
        device_client = sdk_gen2.DeviceClient()
        config        = sdk_gen2.DeviceClientConfig()
        device_client.set_client_config(config)
        device = device_client.connect()
        print("Aria Gen 2 connected!")

        # Configure streaming with profile8 (all sensors)
        streaming_config = sdk_gen2.HttpStreamingConfig()
        streaming_config.profile_name = 'profile8'
        # Use default streaming interface (USB)
        device.set_streaming_config(streaming_config)
        device.start_streaming()

        mode_txt[0] = 'ARIA GEN 2 — Streaming'
        print("Glasses streaming audio.")

        SR=48000; WIN=SR*2; HOP=SR
        buf=[]; held=[0]; st=time.time()

        def audio_callback(audio_data: AudioData,
                           audio_record: AudioDataRecord,
                           num_channels: int):
            try:
                raw = np.array(audio_data.data, dtype=np.float32)
                # Reshape to (channels, samples)
                n = len(raw) // num_channels
                s = raw[:n*num_channels].reshape(num_channels, n)
                mic_lv[0] = float(np.max(np.abs(s)))
                buf.append(s); held[0] += s.shape[-1]
                if held[0] >= WIN:
                    w = np.concatenate(buf, axis=-1)[:, :WIN]
                    buf.clear(); buf.append(w[:, -HOP:])
                    held[0] = HOP
                    threading.Thread(
                        target=run,
                        args=(w[0].astype(np.float32), st, 'GLASSES'),
                        daemon=True).start()
            except Exception as e:
                print(f"  Audio frame error: {e}")

        # Set up stream receiver
        server_config = sdk_gen2.HttpServerConfig()
        server_config.address = '0.0.0.0'
        server_config.port    = 6768

        stream_recv = receiver.StreamReceiver(
            enable_image_decoding=False,
            enable_raw_stream=False)
        stream_recv.set_server_config(server_config)
        stream_recv.register_audio_callback(audio_callback)
        stream_recv.start_server()

        glasses_ok = True
        print("Audio receiver started. Play sounds near the glasses.")
        while True: time.sleep(1)

    except ImportError as e:
        print(f"Aria Gen 2 SDK not available: {e} — falling back to laptop mic.")
    except Exception as e:
        print(f"Aria connection failed: {type(e).__name__}: {e}")
        print("  Falling back to laptop mic.")

    if glasses_ok:
        return

    # Step 3: try laptop mic
    try:
        import sounddevice as sd
        SR=32000; WIN=SR*2; HOP=SR
        buf=[]; held=[0]; st=time.time()
        mode_txt[0] = 'LAPTOP MIC — Listening'
        print(f"Laptop mic at {SR}Hz. Play sounds near laptop.")

        def cb(indata, frames, t_info, status):
            mic_lv[0] = float(np.max(np.abs(indata)))
            mono = indata[:,0].copy()
            buf.append(mono); held[0]+=len(mono)
            if held[0]>=WIN:
                audio=np.concatenate(buf)[:WIN]
                buf.clear(); buf.append(audio[-HOP:])
                held[0]=HOP
                threading.Thread(
                    target=run,
                    args=(audio,st,'MIC'),
                    daemon=True).start()

        with sd.InputStream(samplerate=SR, channels=1,
                            dtype='float32',
                            blocksize=int(SR*0.1),
                            callback=cb):
            print("Mic stream open.")
            while True: time.sleep(0.1)

    except ImportError:
        print("sounddevice not installed.")
        mode_txt[0] = 'DEMO MODE — pip install sounddevice'
    except Exception as e:
        print(f"Mic error: {e}")
        mode_txt[0] = f'MIC ERROR — press 1-5 for demo'

# ── Start audio in background ─────────────────────────────────────
if args.vrs:
    mode_txt[0] = f'VRS PLAYBACK — {Path(args.vrs).name}'
    t = threading.Thread(
        target=vrs_playback_thread,
        args=(args.vrs, args.room or 'kitchen'),
        daemon=True)
    t.start()
elif args.demo:
    mode_txt[0] = 'DEMO MODE — Press 1-5'
    print("Demo mode.")
else:
    threading.Thread(target=audio_thread, daemon=True).start()

# ── Pygame — starts immediately, no waiting ───────────────────────
pygame.init()
if args.fullscreen:
    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
else:
    screen = pygame.display.set_mode((1100,650), pygame.RESIZABLE)
pygame.display.set_caption('ESAS — Environmental Sound Alerting System')
clock = pygame.time.Clock()

fH = pygame.font.SysFont('Arial', 72, bold=True)
fB = pygame.font.SysFont('Arial', 50, bold=True)
fM = pygame.font.SysFont('Arial', 34, bold=True)
fS = pygame.font.SysFont('Arial', 24)
fX = pygame.font.SysFont('Arial', 17)

DARK  = (8,  10, 22)
PANEL = (28, 32, 50)
WHITE = (255,255,255)
GREY  = (90, 90,110)
BLUE  = (50,110,220)

t0      = time.time()
pulse   = 0.0
running = True
print("Window open — click it first, then press 1-5.")

while running:
    now  = time.time()
    W, H = screen.get_size()

    # events
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        if ev.type == pygame.KEYDOWN:
            k = pygame.key.name(ev.key)
            if k == 'escape':
                running = False
            elif k == 'f':
                pygame.display.toggle_fullscreen()
            elif k in MANUAL:
                s,p,d,r,c = MANUAL[k]
                push(s, p, d, r, c, 'KEY')

    # drain queue — this is how alerts appear on screen
    try:
        item       = alert_q.get_nowait()
        current[0] = item
        show_t[0]  = now + 5.5
        history.append(item)
        if len(history) > 6: history.pop(0)
    except queue.Empty:
        pass

    if current[0] and now > show_t[0]:
        current[0] = None

    pulse = (pulse + 0.04) % 1.0
    pv    = abs(pygame.math.Vector2(1,0).rotate(pulse*360).x)

    # ── DRAW ─────────────────────────────────────────────────────
    if current[0]:
        sound,priority,direction,room,conf,color,source = current[0]
        screen.fill(DARK)

        # banner
        pygame.draw.rect(screen, color, (0,0,W,70))
        t = fM.render(f'{priority} PRIORITY', True, WHITE)
        screen.blit(t, t.get_rect(center=(W//2,35)))

        # source badge
        sc = BLUE if source in ('GLASSES','MIC') else GREY
        pygame.draw.rect(screen, PANEL, (W-130,10,120,26), border_radius=8)
        t = fX.render(source, True, sc)
        screen.blit(t, t.get_rect(center=(W-70,23)))

        # divider
        pygame.draw.line(screen, PANEL, (W//2,78),(W//2,H-45),2)

        # LEFT — sound icon + name
        cx = W//4; cy = H//2-20
        r  = 65 + int(10*pv)
        pygame.draw.circle(screen, color, (cx,cy), r)
        pygame.draw.circle(screen, WHITE, (cx,cy), r, 3)
        t = fH.render('!', True, WHITE)
        screen.blit(t, t.get_rect(center=(cx,cy)))
        t = fB.render(sound, True, WHITE)
        screen.blit(t, t.get_rect(center=(cx, cy+r+28)))

        # confidence bar
        bw = W//2-80
        pygame.draw.rect(screen, PANEL, (40,cy+r+65,bw,14), border_radius=5)
        fw = max(0, int(bw*conf))
        if fw: pygame.draw.rect(screen, color, (40,cy+r+65,fw,14), border_radius=5)
        t = fX.render(f'{int(conf*100)}%  confidence', True, GREY)
        screen.blit(t, (40, cy+r+84))

        # room
        pygame.draw.rect(screen, PANEL, (40,cy+r+106,bw,32), border_radius=8)
        t = fS.render(f'Room: {room.title() if room else "Detected"}', True, GREY)
        screen.blit(t, t.get_rect(center=(cx, cy+r+122)))

        # RIGHT — direction arrow + label
        cx2 = W*3//4
        if source == 'GLASSES' and direction != 'UNKNOWN':
            arr = ARROWS.get(direction, direction)
            t   = fB.render(arr, True, color)
            screen.blit(t, t.get_rect(center=(cx2, H//2-25)))
            t = fM.render(direction, True, WHITE)
            screen.blit(t, t.get_rect(center=(cx2, H//2+45)))
        else:
            # Laptop mic is mono — no direction available
            t = fM.render('DETECTED', True, color)
            screen.blit(t, t.get_rect(center=(cx2, H//2-40)))
            t = fS.render('Connect Aria Gen 2', True, GREY)
            screen.blit(t, t.get_rect(center=(cx2, H//2+10)))
            t = fS.render('for direction sensing', True, GREY)
            screen.blit(t, t.get_rect(center=(cx2, H//2+45)))

        # countdown bar
        age = now-(show_t[0]-5.5)
        bw2 = int((W-40)*max(0,1-age/5.5))
        pygame.draw.rect(screen, color, (20,H-32,bw2,8), border_radius=4)

    else:
        screen.fill(DARK)

        # title
        t = fB.render('ESAS', True, BLUE)
        screen.blit(t, t.get_rect(center=(W//2, H//2-90)))
        t = fS.render('Environmental Sound Alerting System', True, GREY)
        screen.blit(t, t.get_rect(center=(W//2, H//2-42)))
        t = fS.render('For Deaf and Hard-of-Hearing Users', True, GREY)
        screen.blit(t, t.get_rect(center=(W//2, H//2-12)))

        # mode pill
        ml = mode_txt[0]
        mc = BLUE if any(x in ml for x in
                         ('ARIA','MIC','LAPTOP')) else (200,130,20)
        pygame.draw.rect(screen, PANEL,
                         (W//2-210,H//2+18,420,36), border_radius=10)
        pygame.draw.rect(screen, mc,
                         (W//2-210,H//2+18,420,36),
                         width=2, border_radius=10)
        t = fX.render(ml, True, mc)
        screen.blit(t, t.get_rect(center=(W//2, H//2+36)))

        # VU bar
        lv = mic_lv[0]
        if lv > 0:
            pygame.draw.rect(screen, PANEL,
                             (W//2-210,H//2+66,420,12), border_radius=4)
            fw2 = min(420, int(420*lv*8))
            if fw2:
                vc = (210,40,40) if lv>.08 else (200,120,20)
                pygame.draw.rect(screen, vc,
                                 (W//2-210,H//2+66,fw2,12), border_radius=4)

        # pulse circle
        r2 = 26+int(7*pv)
        pygame.draw.circle(screen,
            tuple(int(c*(0.3+0.7*pv)) for c in BLUE),
            (W//2, H//2+115), r2, 3)
        t = fS.render('Listening...', True, GREY)
        screen.blit(t, t.get_rect(center=(W//2, H//2+158)))

        # history sidebar
        if history:
            t = fX.render('Recent:', True, GREY)
            screen.blit(t, (W-220, 18))
            for i, h in enumerate(history[-5:][::-1]):
                pygame.draw.rect(screen, PANEL,
                                 (W-220,38+i*30,210,26), border_radius=5)
                t = fX.render(f"[{h[6]}] {h[0][:18]}", True, h[5])
                screen.blit(t, (W-216, 42+i*30))

    # footer
    el = int(now-t0)
    t  = fX.render(
        f'ESAS  |  {mode_txt[0]}  |  Alerts: {len(history)}  |  {el}s',
        True, GREY)
    screen.blit(t, t.get_rect(centerx=W//2, y=H-16))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
print("Done.")
