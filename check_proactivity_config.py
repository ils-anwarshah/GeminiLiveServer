from google.genai import types
print("ProactivityConfig fields:")
try:
    print(types.ProactivityConfig.model_fields.keys())
except:
    print(dir(types.ProactivityConfig))
