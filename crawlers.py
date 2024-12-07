import time
from data.scrapper import VietLottScrapper

def run():
    with VietLottScrapper() as scrapper:
        scrapper.run()

if __name__ =="__main__":
    import time
    start = time.perf_counter()
    run() # start

    elapsed = time.perf_counter() - start
    print(f"File executed in {elapsed:0.2f} seconds")
