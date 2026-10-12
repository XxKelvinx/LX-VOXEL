#pragma once
// The public build exposes no research-mode environment overrides.
inline const char* lxvoxel_release_env(const char*) noexcept { return nullptr; }
