from app.policies.profile_loader import clear_cache, list_profiles, load_profile


def test_load_default_profile():
    clear_cache()
    profile = load_profile("default")
    assert profile["pii_policy"] == "redact"
    assert "medical" in profile["forced_verified_domains"]

def test_load_customer_support_profile():
    clear_cache()
    profile = load_profile("customer_support")
    assert profile["hallucination_check"] == "strict"
    assert profile["complexity_fast_max"] == 3
    # Should inherit default pii_policy (or similar)
    assert profile["pii_policy"] == "redact"

def test_unknown_profile_falls_back_to_default():
    clear_cache()
    default_profile = load_profile("default")
    unknown_profile = load_profile("non_existent_profile_123")
    assert unknown_profile == default_profile

def test_list_profiles():
    clear_cache()
    profiles = list_profiles()
    assert "default" in profiles
    assert "customer_support" in profiles
    assert "internal_knowledge" in profiles
