import json
from pathlib import Path

data_file = Path("data/synthetic_output/generated_data.json")
with open(data_file) as f:
    data = json.load(f)

stats = {}
total_records = 0

for collection_name, records in data.items():
    count = len(records)
    total_records += count
    
    if records:
        first_record = records[0]
        fields = list(first_record.keys())
        
        numeric_fields = {}
        string_fields = {}
        
        for field in fields:
            values = [r.get(field) for r in records if field in r]
            
            # Numeric stats
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                numeric_fields[field] = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values)
                }
            
            # String stats
            string_values = [v for v in values if isinstance(v, str)]
            if string_values:
                unique_values = set(string_values)
                string_fields[field] = {
                    "unique_count": len(unique_values),
                    "sample_values": list(unique_values)[:5]
                }
        
        stats[collection_name] = {
            "count": count,
            "fields": fields,
            "numeric_stats": numeric_fields,
            "string_stats": string_fields
        }

print("📊 Data Statistics")
print("=" * 60)
print(f"Total records: {total_records:,}")
print()

for collection, info in stats.items():
    print(f"📦 {collection.upper()}")
    count = info['count']
    print(f"   Records: {count:,}")
    print(f"   Fields: {len(info['fields'])}")
    field_names = ", ".join(info["fields"])
    print(f"   Field names: {field_names}")
    
    if info["numeric_stats"]:
        print("   Numeric fields:")
        for field, fstats in info["numeric_stats"].items():
            fmin = fstats["min"]
            fmax = fstats["max"]
            fmean = fstats["mean"]
            print(f"      • {field}: min={fmin:.2f}, max={fmax:.2f}, mean={fmean:.2f}")
    
    if info["string_stats"]:
        print("   String fields:")
        for field, fstats in info["string_stats"].items():
            unique = fstats["unique_count"]
            samples = ", ".join(str(v) for v in fstats["sample_values"][:3])
            print(f"      • {field}: {unique} unique values (e.g., {samples})")
    print()
