"""Pure review policy from frozen evidence; never changes recognition/assignment."""


def native_views(output):
    """First hypothesis per view is native Top-1; alternatives are not errors."""
    seen=set()
    for view in output.get("quality",{}).get("view_results",[]):
        key=view.get("view")
        if key not in seen:
            seen.add(key)
            yield view


def apply_review(proposal,maximum):
    from .base import numeric
    p=proposal;reasons=[]
    if p["state"]=="UNCERTAIN":reasons.append("state_uncertain")
    if p["state"]!="UNOPENED":
        if not p["ensemble"]["candidates"]:reasons.append("ocr_no_result")
        if not p["ensemble"]["agreement"]:reasons.append("engines_disagree")
        if p["confidence"]<.65:reasons.append("confidence_low")
        if p.get("corrected"):reasons.append("constraint_corrected")
        engines=[e for e in p.get("raw_engines",[]) if e.get("quality",{}).get("runtime_policy")!="benchmark_only"]
        tops=[e["candidates"][0] for e in engines if e["candidates"]]
        if any(v["score"]>=.8 and v["value"]!=p["value"] for v in tops):
            reasons.append("high_confidence_conflict")
        variants={numeric(v.get("raw_text",""),maximum) for e in engines
                  for v in native_views(e) if v.get("score",0)>=.8}
        variants.discard(None)
        if len(variants)>1:reasons.append("conflicting_confident_views")
        if p.get("value") is not None:
            text=str(p["value"])
            # Even low-scoring native observations are useful review evidence.
            # Do not turn them into corrections or hardcode any missing digit.
            if any(len(str(c["value"]))>len(text) and text in str(c["value"])
                   for c in p["ensemble"]["candidates"]):
                reasons.append("potentially_truncated_number")
        quality=p.get("quality",{})
        if quality.get("glare",0)>.025 and quality.get("brightness",0)>.8:
            reasons.append("specular_glare")
    p["review_reasons"]=reasons
    p["disputed"]=bool(reasons)
    p["needs_confirmation"]=bool(reasons)
    return p
