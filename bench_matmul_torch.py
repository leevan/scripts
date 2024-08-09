import torch
import time
import torch.profiler

print('Pytorch version\t:', torch.__version__)
print('CUDA version\t:', torch.version.cuda)
print('GPU\t\t:',torch.cuda.get_device_name())

m = 8192 #4096 #1024
k = 8192 #5120 #8192
n = 8192 #15360 #8192
for dtype in (torch.float32, torch.float16):
    a = torch.randn(m, k, dtype=dtype).cuda()
    b = torch.randn(k, n, dtype=dtype).cuda()

    st = time.time()
    for i in range(100):
        c = torch.matmul(a, b)

        torch.cuda.synchronize()
    et = time.time()
    t = (et - st) / 100
    print(t)
    print("flops : {0:4.4f}".format(2*m*n*k / t / 1e12))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA],
                           ) as prof:
        for _ in range(10):
            #prof.step()
            c = torch.matmul(a,b)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #prof.export_chrome_trace("mm_trace_sync.json")
