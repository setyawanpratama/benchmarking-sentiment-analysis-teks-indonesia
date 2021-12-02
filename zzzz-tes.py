import os

# DATA_TEMPLATED = ["./Dataset/templated/" + x for x in os.listdir("./Dataset/templated")]
DATA_TEMPLATED = [x for x in os.listdir("./Dataset/templated")]
DATA_TEMPLATED.sort
# DATA_CLEAN = ["./Dataset/clean/" + x for x in os.listdir("./Dataset/clean")]
DATA_CLEAN = [x for x in os.listdir("./Dataset/clean")]
DATA_CLEAN.sort

print(len(DATA_TEMPLATED))
print(len(DATA_CLEAN))
for i in range(max(len(DATA_CLEAN), len(DATA_TEMPLATED))):
    print(DATA_TEMPLATED[i], " | ", DATA_CLEAN[i])