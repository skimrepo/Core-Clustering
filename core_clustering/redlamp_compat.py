"""Constants needed to train a model here that is later loadable, with
semantically matching classifier output, into RedLamp's own ConvAEC.

RedLamp's main.anomaly_scoreing() hardcodes "classifier output index 0 is
the 'normal' class" (RedLamp/main.py, label_score_selected_feature always
forces index 0 into the excluded/zeroed set). A model trained here with a
differently-ordered class_list would still load (state_dict shapes only
depend on class *count*), but its channel semantics would silently no
longer mean what RedLamp's scoring code assumes -- producing a garbage,
not-obviously-wrong anomaly score. Pass this list as load_windowed_dataset's
class_list= to avoid that.
"""

REDLAMP_ANOMALY_TYPES = [
    "normal", "spike", "flip", "speedup", "noise", "cutoff",
    "average", "scale", "wander", "contextual", "upsidedown", "mixture",
]
