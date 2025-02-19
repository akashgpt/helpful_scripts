#write a file to generate numbers in serial order between 1 and 500 in a file called test-frame-select-fps-n-500.index

# take num_frames as input from cmd line
import sys
num_frames = sys.argv[1]
num_frames = int(num_frames)


# Open the file in write mode
with open(f'test-frame-select-fps-n-{num_frames}.index', 'w') as file:
    # Write numbers from 1 to 500, each on a new line
    for number in range(0, num_frames):
        file.write(f"{number}\n")