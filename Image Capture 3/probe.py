# probe.py
import time
from arena_api.system import system

infos = system.device_infos
dev = system.create_device([infos[0]])[0]
nm, snm = dev.nodemap, dev.tl_stream_nodemap

snm['StreamBufferHandlingMode'].value      = 'NewestOnly'
snm['StreamAutoNegotiatePacketSize'].value = True
snm['StreamPacketResendEnable'].value      = True
nm['DeviceStreamChannelPacketSize'].value  = 1500         # match a 1500 MTU NIC
nm['PixelFormat'].value = 'RGB8'

# --- the two diagnostics that matter ---
print("negotiated packet size:", nm['DeviceStreamChannelPacketSize'].value)
try:
    print("exposure auto:", nm['ExposureAuto'].value)
    print("exposure time :", nm['ExposureTime'].value, "us")
except Exception as e:
    print("exposure read failed:", e)

dev.start_stream()
print("waiting up to 15s for one frame ...")
t0 = time.time()
try:
    b = dev.get_buffer(timeout=15000)
    print(f"GOT FRAME {b.width}x{b.height} after {time.time()-t0:.2f}s")
    dev.requeue_buffer(b)
except Exception as e:
    print(f"STILL TIMED OUT after {time.time()-t0:.2f}s:", repr(e))

dev.stop_stream()
system.destroy_device()