---
name: cleanup-validation
description: Validate one image-cleanup candidate and its paired description against the raw image.
---

# Cleanup Validation

Evaluate only the current attempt using the raw source image, the current candidate image, its paired object description, and its attempt index. Do not generate or edit an image, author a new description, create a prompt, compare attempts, or select a final result.

Return exactly one record:

```json
{
  "attempt_index": 1,
  "verdict": "pass | fail",
  "checks": {
    "object_identity": {"passed": true, "evidence": "..."},
    "single_complete_instance": {"passed": true, "evidence": "..."},
    "silhouette_proportions_and_openings": {"passed": true, "evidence": "..."},
    "part_boundaries_joints_and_compliance": {"passed": true, "evidence": "..."},
    "material_cues": {"passed": true, "evidence": "..."},
    "pose_and_visibility": {"passed": true, "evidence": "..."},
    "background_and_artifacts": {"passed": true, "evidence": "..."},
    "description_alignment": {"passed": true, "evidence": "..."}
  },
  "summary": "short current-attempt judgment"
}
```

Each evidence field is non-empty. `verdict` is `pass` exactly when every check passes. Treat the raw image as the identity and geometry baseline; preserve visible functional structure over presentation polish.
Require a spatially uniform pure-white background with no shadows or lighting effects. Any non-white or non-uniform background, shadow, or lighting effect makes `background_and_artifacts.passed` false.
