import csv
import psutil

def write_to_csv(cpu_percent):
    with open('cpu.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cpu_percent])

# Get current CPU percentage
while True:
    cpu_percent = psutil.cpu_percent(interval = 1)

    # Write to CSV file
    write_to_csv(cpu_percent)
