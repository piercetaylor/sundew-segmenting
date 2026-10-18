// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// Switches for parts of the site that are built but not shipped.
//
// OUTLINE: the plant outline drawn over the photo (index.html: #outline canvas, "Show plant outline"
// toggle, #outline-area readout). Off until a segmentation model passes its pre-registered bar: the
// U-Net failed it (docs/reports/segmentation-site.md), and a DINOv2-S head is next. While false,
// nothing about the outline loads or shows, and the page makes no extra requests. Turning it on also
// needs identify.js to run the segmentation model on each photo, draw the mask on #outline, and show
// the plant's share of the photo in #outline-area (resetOutline() clears them for a new photo).
export const OUTLINE = false;
