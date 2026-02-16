from google.genai import types
print("LiveConnectConfig fields:")
try:
    print(types.LiveConnectConfig.model_fields['proactivity'])
except:
    pass

print("\nSearching for Proactivity types:")
for name in dir(types):
    if 'Proactivity' in name:
        print(name)
