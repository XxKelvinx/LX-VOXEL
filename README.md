![LX-VOXEL](LX-VOXEL-Logo.png)

## Introduction

**LX-VOXEL — Low Complexity VOXEL — is an experimental lossy audio codec built around direct lattice vector quantisation (LVQ), fast rANS entropy coding, and an adaptive MDCT with a short overlap.**

The codec combines four block sizes — **1024, 768, 512, and 384 samples** — with a fixed **128-sample overlap**. This provides a balance between frequency resolution, transient handling, and computational cost.

Its perceptual allocation is driven by **FEAR**, the **Fast Ear Auditory Rate-Model**. **FBR**, the **FEAR Bit Rate** system, translates this analysis into a global bit allocation through water-filling: it distributes the available budget across the track according to perceptual importance.

LX-VOXEL brings together several months of research into vector quantisation, transform design, psychoacoustics, and energy preservation. Its central goal is to make these techniques work together in a fast, practical audio codec.

## Research Background

### From PULSAR to VOXEL

LX-VOXEL is the latest outcome of a longer research journey.

The project began with **PULSAR**, an Opus-inspired experiment using pyramid vector quantisation (PVQ), a technique also used in [Opus/CELT](https://www.rfc-editor.org/rfc/rfc6716.html). Although PULSAR did not reach its intended goals, it provided a starting point for exploring different approaches to audio quantisation.

The first **VOXEL** prototypes used scalar quantisation with a trellis search. This produced interesting results at higher bitrates, but the computational cost encouraged further research into alternative quantisers.

### Exploring Lattice Vector Quantisation

The next stage explored LVQ, beginning with the **16-dimensional Barnes–Wall lattice**. Experiments expanded to 32 and 64 dimensions, and eventually included attempts to represent entire frames as single large vectors.

These experiments showed benefits from larger vectors through **space-filling gain, or “room gain,” and more isotropic quantisation error**. A more isotropic error distribution spreads the error more evenly across directions in the vector space. In our experiments, this helped hide quantisation errors perceptually more effectively than scalar quantisation, contributing to better perceived audio quality.

A particularly promising approach was **direct vector quantisation without an explicit shape–gain split**. Instead of normalising a vector and coding its direction and magnitude separately, direct LVQ represents both through the selected lattice point at the chosen quantisation scale.

This allows the quantiser to represent amplitude as part of the vector itself and became one of the central ideas behind VOXEL.

### Short Overlap and Adaptive Blocks

Quantisation was only part of the problem. The transform also had a substantial effect on the character of the reconstructed audio.

In our experiments, longer overlaps could reduce spectral leakage for relatively stationary signals, but could also spread quantisation artefacts further in time. Around transients, this temporal smearing could be more objectionable than the spectral trade-offs of a shorter overlap.

These observations led to the short-overlap MDCT used in LX-VOXEL.

Adaptive block switching was then introduced, with the final block sizes settling at **1024, 768, 512, and 384 samples**. Longer blocks provide finer frequency resolution, while shorter blocks improve temporal localisation.

The 384-sample minimum was a deliberate compromise. In our experiments, it retained useful frequency resolution while handling transients effectively in combination with the short overlap.

### Building a Lower-Complexity Codec

LX-VOXEL emerged from the idea of turning the broader VOXEL research into a simpler, faster implementation.

It uses the **Dₙ lattice** instead of the more complex lattices explored in earlier prototypes. Its simple structure allows quantisation to remain close to coordinate-wise rounding, followed by a parity correction.

Grouping more coefficients into a vector provided a practical way to retain useful vector-quantisation behaviour while keeping the underlying lattice inexpensive to process.

In our experiments, this approach brought the initial quantisation stage close to the speed of scalar quantisation, making it a promising foundation for a codec focused on low complexity.

### FEAR, Bit Allocation, and Energy Stability

A fast quantiser still needs effective psychoacoustics.

**FEAR** was developed and tuned using feedback from [Zimtohrli](https://github.com/google/zimtohrli), a perceptual audio similarity metric. The aim was to reduce perceptually significant errors by directing precision toward the regions where it mattered most.

This research also highlighted the importance of temporal behaviour and energy stability. Minimising coefficient error alone did not consistently produce the most stable perceptual result: changes in reconstructed energy between frames could remain audible.

To address the quantisation error and energy mismatch, LX-VOXEL incorporates **vector refinement that optimises mean squared error (MSE) and energy preservation**. A decoder-side conditional gain correction further adjusts reconstruction energy using information derived from the decoded lattice vectors.

Together, these techniques improved energy stability and perceived consistency in our development tests.

## Results and Findings

The research led to several practical findings:

- **Larger vectors provided perceptual benefits.** Space-filling gain and more isotropic quantisation error improved perceived quality, with isotropy helping hide errors more effectively than scalar quantisation.
- **Simple lattices can be useful foundations for fast audio coding.** The Dₙ lattice offered a workable balance between vector quantisation and computational cost.
- **Short overlap worked well with the selected block sizes.** This combination provided effective transient handling while retaining useful frequency resolution.
- **MSE and energy preservation both matter.** Vector refinement optimises these two objectives, while decoder-side gain correction further improves reconstructed energy.
- **Global allocation connects the individual components.** FEAR supplies the perceptual priorities, while FBR distributes the track’s available bit budget across them.

The resulting implementation includes command-line encoding and decoding tools, metadata support, and a reusable decoder exposed through a C API. CPU dispatch provides an SSE4.1 baseline and an AVX2/FMA path on x86, with a NEON implementation for ARM.

These findings describe the project’s development experience; quantitative comparisons require published benchmark conditions and results.

## Use Cases

**Offline audio encoding**

The default global allocation approach is suited to file encoding, where the encoder can analyse the complete track before distributing its bit budget. The current audio format targets 48 kHz stereo.

**Custom playback applications**

The decoder C API supports integration into players and other applications, including PCM reads, seeking, and access to metadata and embedded cover art.

**Audio codec research**

LX-VOXEL provides an implementation for exploring direct LVQ, perceptual water-filling, adaptive transforms, and vector refinement that optimises MSE and energy preservation.

## Summary

LX-VOXEL brings the broader VOXEL research into a codec built around a computationally simple lattice and a carefully coordinated perceptual pipeline.

Larger vectors offered benefits through space-filling gain and isotropic error behaviour. Short overlap and adaptive blocks shaped how those errors behaved in time, while FEAR and FBR directed precision toward perceptually important regions. Vector refinement and decoder-side gain correction helped preserve reconstruction energy.

LX-VOXEL combines these lessons into a practical implementation for further experimentation and integration.

## License

LX-VOXEL is distributed under the **LX-VOXEL Source License 1.2**. See [LICENSE](https://github.com/XxKelvinx/LX-VOXEL/blob/main/LICENSE) for the applicable terms.

## Acknowledgements and AI Assistance

LX-VOXEL was developed with the help of **GPT 5.6 Sol**.

This README was also written and edited with AI assistance.
