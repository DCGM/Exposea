import threading
import time
import pynvml
from register import main
import tracemalloc
import logging
import hydra


class GPUMemoryMonitor(threading.Thread):
    def __init__(self, device_index=0, interval=0.1):
        super().__init__()
        self.device_index = device_index
        self.interval = interval
        self.running = False
        self.peak_memory = 0  # in MB

    def run(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self.running = True
        while self.running:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = meminfo.used / 1024 / 1024
            self.peak_memory = max(self.peak_memory, used_mb)
            time.sleep(self.interval)
        pynvml.nvmlShutdown()

    def stop(self):
        self.running = False
        self.join()



@hydra.main(version_base=None, config_path="../configs", config_name="david")
def m_main(config):
    monitor = GPUMemoryMonitor(device_index=0)
    monitor.start()

    try:
        tracemalloc.start()

        main(config)

        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        logging.info(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

    finally:
        monitor.stop()
        print(f"\n🔺 Peak GPU memory usage: {monitor.peak_memory:.2f} MB")

if __name__ == '__main__':
    m_main()