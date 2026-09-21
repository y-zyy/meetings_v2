from diarizen.pipelines.inference import DiariZenPipeline

# load pre-trained model
diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")
# apply diarization pipeline
diar_results = diar_pipeline('audio.wav')

# print results
for turn, _, speaker in diar_results.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# load pre-trained model and save RTTM result
diar_pipeline = DiariZenPipeline.from_pretrained(
        "BUT-FIT/diarizen-wavlm-large-s80-md",
        rttm_out_dir='.'
)
# apply diarization pipeline
diar_results = diar_pipeline('audio.wav', sess_name='session_name')
