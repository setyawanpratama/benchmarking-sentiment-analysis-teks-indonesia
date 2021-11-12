with open("log.csv", "w") as csv:
    csv.write("{},{},{},{}\n".format('a', 1, 2, 3))
csv.close()

with open("log.csv", "a") as csv:
    csv.write("{},{},{},{}\n".format('a', 1, 2, 3))
csv.close()

with open("log.csv", "a") as csv:
    csv.write("{},{},{},{}\n".format('a', 1, 2, 3))
csv.close()

with open("log.csv", "a") as csv:
    csv.write("{},{},{},{}\n".format('a', 1, 2, 3))
csv.close()