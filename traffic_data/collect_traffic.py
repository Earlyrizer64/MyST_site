import sys, json, csv, os
from datetime import datetime

data = json.load(sys.stdin)
file = 'traffic_data/views.csv'
exists = os.path.isfile(file)

# Read existing dates to avoid duplicates
existing_dates = set()
if exists:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                existing_dates.add(row[0])

with open(file, 'a', newline='') as f:
    writer = csv.writer(f)
    if not exists:
        writer.writerow(['date', 'views', 'unique_visitors'])
    for entry in data.get('views', []):
        date = entry['timestamp'][:10]
        if date not in existing_dates:
            writer.writerow([date, entry['count'], entry['uniques']])
            existing_dates.add(date)
