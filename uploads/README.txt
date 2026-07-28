OCR READINESS TEST IMAGE SET (20 images / 10 factors x 2 conditions each)
===========================================================================

Filename                         Factor                         Condition                  Script
--------------------------------------------------------------------------------------------------------------
01_noise_low_english.png         Noise                          LOW noise                  English
02_noise_high_hindi.png          Noise                          HIGH noise                 Hindi
03_resolution_low_mixed.png      Resolution                     LOW resolution             Mixed (English+Hindi)
04_resolution_high_english.png   Resolution                     HIGH resolution            English
05_blur_low_sharp_hindi.png      Blur                           LOW blur (sharp)           Hindi
06_blur_high_mixed.png           Blur                           HIGH blur                  Mixed (English+Hindi)
07_contrast_low_english.png      Contrast                       LOW contrast               English
08_contrast_high_hindi.png       Contrast                       HIGH contrast              Hindi
09_stroke_thin_mixed.png         Stroke Width                   THIN strokes               Mixed (English+Hindi)
10_stroke_thick_english.png      Stroke Width                   THICK strokes              English
11_density_low_hindi.png         Text Density                   LOW density (sparse)       Hindi
12_density_high_mixed.png        Text Density                   HIGH density (crowded)     Mixed (English+Hindi)
13_matra_good_hindi.png          Matra Continuity               GOOD continuity            Hindi
14_matra_broken_hindi.png        Matra Continuity               BROKEN continuity          Hindi
15_zone_good_hindi.png           Zone Integrity                 GOOD (all zones intact)    Hindi
16_zone_poor_mixed.png           Zone Integrity                 POOR (clipped zones)       Mixed (English+Hindi)
17_ccstability_good_english.png  Connected Component Stability  GOOD (clean)               English
18_ccstability_poor_mixed.png    Connected Component Stability  POOR (fragmented/noisy)    Mixed (English+Hindi)
19_skew_low_english.png          Skew                           LOW skew (straight)        English
20_skew_high_hindi.png           Skew                           HIGH skew (rotated)        Hindi


DETAILS
===========================================================================

01_noise_low_english.png
  Factor: Noise
  Condition: LOW noise
  Script: English
  Notes: Clean English text with only light sensor-level noise (sigma=4).

02_noise_high_hindi.png
  Factor: Noise
  Condition: HIGH noise
  Script: Hindi
  Notes: Hindi text with heavy Gaussian noise (sigma=40) simulating a poor low-light photo.

03_resolution_low_mixed.png
  Factor: Resolution
  Condition: LOW resolution
  Script: Mixed (English+Hindi)
  Notes: Mixed-script document downscaled to a tiny crop (~180x90px) simulating a low-res capture.

04_resolution_high_english.png
  Factor: Resolution
  Condition: HIGH resolution
  Script: English
  Notes: English document upscaled to a large high-resolution canvas (3600x1800px).

05_blur_low_sharp_hindi.png
  Factor: Blur
  Condition: LOW blur (sharp)
  Script: Hindi
  Notes: Perfectly sharp, crisp Hindi text with no blur applied.

06_blur_high_mixed.png
  Factor: Blur
  Condition: HIGH blur
  Script: Mixed (English+Hindi)
  Notes: Mixed-script document heavily Gaussian-blurred (kernel=21) simulating an out-of-focus photo.

07_contrast_low_english.png
  Factor: Contrast
  Condition: LOW contrast
  Script: English
  Notes: English text rendered in faded gray-on-light-gray to simulate washed-out low contrast.

08_contrast_high_hindi.png
  Factor: Contrast
  Condition: HIGH contrast
  Script: Hindi
  Notes: Crisp pure black-on-white Hindi text with maximum contrast.

09_stroke_thin_mixed.png
  Factor: Stroke Width
  Condition: THIN strokes
  Script: Mixed (English+Hindi)
  Notes: Mixed-script document rendered at a small font size producing very thin (~1px) strokes.

10_stroke_thick_english.png
  Factor: Stroke Width
  Condition: THICK strokes
  Script: English
  Notes: Bold, heavily-stroked English text simulating an overly thick/bleeding print.

11_density_low_hindi.png
  Factor: Text Density
  Condition: LOW density (sparse)
  Script: Hindi
  Notes: A single short line of Hindi text on a large mostly-blank page (very sparse ink coverage).

12_density_high_mixed.png
  Factor: Text Density
  Condition: HIGH density (crowded)
  Script: Mixed (English+Hindi)
  Notes: Tightly packed mixed-script lines with minimal line spacing, simulating dense/crowded text.

13_matra_good_hindi.png
  Factor: Matra Continuity
  Condition: GOOD continuity
  Script: Hindi
  Notes: Clean printed Hindi text with an unbroken Shirorekha (headline) across each word.

14_matra_broken_hindi.png
  Factor: Matra Continuity
  Condition: BROKEN continuity
  Script: Hindi
  Notes: Same Hindi text with the Shirorekha headline artificially broken in many places, simulating scan damage.

15_zone_good_hindi.png
  Factor: Zone Integrity
  Condition: GOOD (all zones intact)
  Script: Hindi
  Notes: Full Hindi text with upper (matra), middle (body) and lower (descender) zones all intact.

16_zone_poor_mixed.png
  Factor: Zone Integrity
  Condition: POOR (clipped zones)
  Script: Mixed (English+Hindi)
  Notes: Mixed-script text with the upper vowel-sign zone and part of the lower zone clipped off (bad crop).

17_ccstability_good_english.png
  Factor: Connected Component Stability
  Condition: GOOD (clean)
  Script: English
  Notes: Clean, well-formed English text with consistent, undamaged character shapes.

18_ccstability_poor_mixed.png
  Factor: Connected Component Stability
  Condition: POOR (fragmented/noisy)
  Script: Mixed (English+Hindi)
  Notes: Mixed-script text with random speckle noise and many small cuts fragmenting the characters.

19_skew_low_english.png
  Factor: Skew
  Condition: LOW skew (straight)
  Script: English
  Notes: Perfectly horizontal, unrotated English text.

20_skew_high_hindi.png
  Factor: Skew
  Condition: HIGH skew (rotated)
  Script: Hindi
  Notes: Hindi text rotated by 22 degrees simulating a tilted photo capture.
