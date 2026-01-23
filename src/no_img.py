import os

test_normal = len(os.listdir("data/chest_xray/test/NORMAL"))
test_pneumonia = len(os.listdir("data/chest_xray/test/PNEUMONIA"))

print("Test NORMAL:", test_normal)
print("Test PNEUMONIA:", test_pneumonia)