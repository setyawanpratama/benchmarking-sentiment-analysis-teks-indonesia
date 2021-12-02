import time


class Timer:
    def __init__(self, description):
        self.start_time = time.time()
        self.description = description
        print("{} started ...".format(description))

    def stop(self) -> dict:
        self.end_time = time.time()
        time_diff = self.end_time - self.start_time
        hours = int(time_diff // 3600)
        minutes = int((time_diff - (hours * 3600)) // 60)
        seconds = round(time_diff - (hours * 3600) - (minutes * 60), 3)
        status = "Elapsed time for {:<30} = {} hours, {} minutes, {} seconds".format(
            self.description, hours, minutes, seconds
        )
        print("{:<150}".format(status))

        return {
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds
        }
