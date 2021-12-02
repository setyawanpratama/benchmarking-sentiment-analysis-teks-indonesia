import os

class CSVLogger():
    def __init__(self, file) -> str:
        self.file = file
        with open(self.file, "w") as csv:
            csv.write("")
        csv.close()
    
    def do_log(self, item:list):
        item_text = ",".join(item)
        with open(self.file, "a") as csv:
            csv.write(item_text + "\n")
        csv.close()