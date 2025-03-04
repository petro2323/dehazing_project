import qai_hub as hub

profile_job = hub.submit_profile_job(
    model="job_jpvqe28rg_optimized_onnx_mngry295q.onnx",
    device=hub.Device("QCS8550 (Proxy)"),
)
assert isinstance(profile_job, hub.ProfileJob)