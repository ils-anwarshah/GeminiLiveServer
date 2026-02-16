from google.genai import types
print("ProactivityConfig.proactive_audio type:")
try:
    print(types.ProactivityConfig.model_fields['proactive_audio'])
except:
    pass
