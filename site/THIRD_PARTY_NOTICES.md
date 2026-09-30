# Third-party notices

The code of this site (`index.html`, `notice.html`, `classify.js`, `pilresize.js`, `js/decision.js`) is
licensed under the Apache License, Version 2.0 (see the repository's
`LICENSE` and `NOTICE`). The model weights it downloads
(`model/v1.0.0/model-int8.onnx`) are licensed under CC BY-NC 4.0; see
`release/species-v1.0.0/README.md` and `ATTRIBUTION.md` in the repository.

The site includes or uses the following third-party work.

## Pillow (MIT-CMU licence, also known as HPND)

`pilresize.js` is a JavaScript port of the image resampling code in Pillow's
`src/libImaging/Resample.c`, so that the browser resizes photos exactly as the
model's training did.

```text
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

## ONNX Runtime Web (MIT licence)

The site serves three files of `onnxruntime-web` 1.30.0
(<https://www.npmjs.com/package/onnxruntime-web>) from `ort/1.30.0/`, copied
unmodified from the npm package when the site is built. Only the optional
`?backend=webgpu` mode loads its WebGPU build from jsDelivr instead.

```text
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DINOv2 (Apache License 2.0)

The model is a DINOv2-S image classifier (Meta AI, via timm), fine-tuned and
distilled for sundew species. DINOv2 is Copyright (c) Meta Platforms, Inc. and
affiliates, licensed under the Apache License, Version 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>), the same licence text as the
repository's `LICENSE`. See the model card for details.
