import shutil
import sys
import time

def progress_bar(iterable, length=100):
    total = len(iterable)
    progress = 0
    start_time = time.time()

    
    length = length - 12  # Adjusting for other characters in the progress bar
  


    for i, item in enumerate(iterable):
        yield item
        progress += 1
        elapsed_time = time.time() - start_time
        progress_percent = progress / total
        progress_length = int(length * progress_percent)
        remaining_length = length - progress_length

       
        progress_bar = '█' * progress_length + '-' * remaining_length   # options: '█', '=', etc.
        sys.stdout.write(f'\r[{progress_bar}] {progress}/{total} ({progress_percent:.1%}), Elapsed Time: {elapsed_time:.2f}s')
        sys.stdout.flush()
    sys.stdout.write('\n')

# Example usage
data = range(100)
for _ in progress_bar(data):
    time.sleep(0.1)  # Simulate some processing time

print("done!")

