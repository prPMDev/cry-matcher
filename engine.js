// Cry Matcher engine: YAMNet log-mel features + network in TF.js, and matching.
// Mirrors cry_match.py (YAMNet embeddings, cry-window filter, centered cosine to centroids).
(function (root) {
  "use strict";
  var SR = 16000, WIN = 400, HOP = 160, NFFT = 512, NBIN = 257, MELS = 64;
  var PATCH = 96, PATCH_HOP = 48, MIN_SAMPLES = 15600, HOP_SAMPLES = 7680;
  var CRY_THRESHOLD = 0.10;
  var DEFS = [[2, 32], [1, 64], [2, 128], [1, 128], [2, 256], [1, 256], [2, 512],
    [1, 512], [1, 512], [1, 512], [1, 512], [1, 512], [2, 1024], [1, 1024]];

  function f16to32(u16) {
    var out = new Float32Array(u16.length);
    for (var i = 0; i < u16.length; i++) {
      var h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * f * 5.960464477539063e-8
        : e === 31 ? (f ? NaN : s * Infinity)
        : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }

  function b64ToBytes(b64) {
    if (typeof atob === "function") {
      var bin = atob(b64), n = bin.length, out = new Uint8Array(n);
      for (var i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(b64, "base64"));
  }

  // Periodic Hann window, as tf.signal.stft uses.
  var HANN = new Float32Array(WIN);
  for (var i = 0; i < WIN; i++) HANN[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / WIN);

  // Radix-2 FFT tables for NFFT.
  var REV = new Uint16Array(NFFT), COS = new Float64Array(NFFT / 2), SIN = new Float64Array(NFFT / 2);
  (function () {
    var bits = Math.log2(NFFT);
    for (var i = 0; i < NFFT; i++) {
      var r = 0;
      for (var b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      REV[i] = r;
    }
    for (var k = 0; k < NFFT / 2; k++) { COS[k] = Math.cos(-2 * Math.PI * k / NFFT); SIN[k] = Math.sin(-2 * Math.PI * k / NFFT); }
  })();

  function fftMag(frame, re, im, mag) {
    for (var i = 0; i < NFFT; i++) { re[REV[i]] = i < WIN ? frame[i] : 0; im[i] = 0; }
    for (var size = 2; size <= NFFT; size <<= 1) {
      var half = size >> 1, step = NFFT / size;
      for (var st = 0; st < NFFT; st += size) {
        for (var j = 0; j < half; j++) {
          var k = j * step, a = st + j, b = a + half;
          var tr = re[b] * COS[k] - im[b] * SIN[k], ti = re[b] * SIN[k] + im[b] * COS[k];
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        }
      }
    }
    for (var m = 0; m < NBIN; m++) mag[m] = Math.sqrt(re[m] * re[m] + im[m] * im[m]);
  }

  function padWaveform(x) {
    var n = Math.max(x.length, MIN_SAMPLES);
    var hops = Math.ceil((n - MIN_SAMPLES) / HOP_SAMPLES);
    var total = MIN_SAMPLES + hops * HOP_SAMPLES;
    if (total === x.length) return x;
    var out = new Float32Array(total); out.set(x); return out;
  }

  // Returns {data: Float32Array(P*96*64), patches: P}
  function logMelPatches(wave, melW) {
    var x = padWaveform(wave);
    var frames = 1 + Math.floor((x.length - WIN) / HOP);
    var logmel = new Float32Array(frames * MELS);
    var fr = new Float32Array(WIN), re = new Float64Array(NFFT), im = new Float64Array(NFFT), mag = new Float64Array(NBIN);
    for (var t = 0; t < frames; t++) {
      var o = t * HOP;
      for (var i = 0; i < WIN; i++) fr[i] = x[o + i] * HANN[i];
      fftMag(fr, re, im, mag);
      for (var m = 0; m < MELS; m++) {
        var s = 0;
        for (var b = 0; b < NBIN; b++) s += mag[b] * melW[b * MELS + m];
        logmel[t * MELS + m] = Math.log(s + 0.001);
      }
    }
    var P = 1 + Math.floor((frames - PATCH) / PATCH_HOP);
    var data = new Float32Array(P * PATCH * MELS);
    for (var p = 0; p < P; p++) data.set(logmel.subarray(p * PATCH_HOP * MELS, (p * PATCH_HOP + PATCH) * MELS), p * PATCH * MELS);
    return { data: data, patches: P };
  }

  function Model(tf, W) {
    var bytes = b64ToBytes(W.b64), T = {}, mel = null;
    W.manifest.forEach(function (e) {
      var n = e.shape.reduce(function (a, b) { return a * b; }, 1);
      var arr = e.dtype === "f16"
        ? f16to32(new Uint16Array(bytes.buffer, bytes.byteOffset + e.offset, n))
        : new Float32Array(bytes.buffer.slice(bytes.byteOffset + e.offset, bytes.byteOffset + e.offset + n * 4));
      if (e.name === "mel") mel = arr; else T[e.name] = tf.tensor(arr, e.shape);
    });
    this.tf = tf; this.T = T; this.mel = mel;
  }

  Model.prototype.run = function (wave) {
    var tf = this.tf, T = this.T, f = logMelPatches(wave, this.mel);
    var out = tf.tidy(function () {
      var x = tf.tensor4d(f.data, [f.patches, PATCH, MELS, 1]);
      DEFS.forEach(function (d, i) {
        var n = i + 1;
        if (n === 1) {
          x = tf.relu(tf.add(tf.conv2d(x, T.l1_conv_w, d[0], "same"), T.l1_conv_b));
        } else {
          x = tf.relu(tf.add(tf.depthwiseConv2d(x, T["l" + n + "_dw_w"], d[0], "same"), T["l" + n + "_dw_b"]));
          x = tf.relu(tf.add(tf.conv2d(x, T["l" + n + "_pw_w"], 1, "same"), T["l" + n + "_pw_b"]));
        }
      });
      var emb = tf.mean(x, [1, 2]);
      var cry = tf.sigmoid(tf.add(tf.matMul(emb, T.cry_w), T.cry_b));
      return [emb, cry];
    });
    var emb = out[0].arraySync(), cry = out[1].arraySync();
    out[0].dispose(); out[1].dispose();
    return { emb: emb, cry: cry };
  };

  // One vector per clip, as cry_match.py does it.
  Model.prototype.embedClip = function (wave) {
    var r = this.run(wave), kept = [];
    var cryScores = r.cry.map(function (c) { return Math.max(c[0], c[1]); });
    cryScores.forEach(function (c, i) { if (c >= CRY_THRESHOLD) kept.push(r.emb[i]); });
    var used = kept.length ? kept : r.emb, v = new Float32Array(1024);
    used.forEach(function (e) { for (var k = 0; k < 1024; k++) v[k] += e[k] / used.length; });
    return { vec: normalize(v), cryShare: kept.length / r.emb.length };
  };

  function normalize(v) {
    var s = 0; for (var i = 0; i < v.length; i++) s += v[i] * v[i];
    s = Math.sqrt(s) + 1e-9;
    var o = new Float32Array(v.length); for (var j = 0; j < v.length; j++) o[j] = v[j] / s;
    return o;
  }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function meanVec(vs) {
    var m = new Float32Array(vs[0].length);
    vs.forEach(function (v) { for (var i = 0; i < m.length; i++) m[i] += v[i] / vs.length; });
    return m;
  }
  function center(v, mean) {
    var o = new Float32Array(v.length); for (var i = 0; i < v.length; i++) o[i] = v[i] - mean[i];
    return normalize(o);
  }

  // refs: [{sound, vec}] (only sounds that are ready). Returns ranked [{sound, score}].
  function match(vec, refs) {
    var mean = meanVec(refs.map(function (r) { return r.vec; }));
    var by = {};
    refs.forEach(function (r) { (by[r.sound] = by[r.sound] || []).push(center(r.vec, mean)); });
    var q = center(vec, mean);
    return Object.keys(by).map(function (s) {
      return { sound: s, score: dot(q, normalize(meanVec(by[s]))) };
    }).sort(function (a, b) { return b.score - a.score; });
  }

  // Leave-one-out check of the references, as `cry_match.py check`.
  function check(refs) {
    var mean = meanVec(refs.map(function (r) { return r.vec; }));
    var c = refs.map(function (r) { return { sound: r.sound, v: center(r.vec, mean) }; });
    var sounds = Array.from(new Set(c.map(function (r) { return r.sound; }))).sort(), res = {};
    sounds.forEach(function (s) { res[s] = [0, 0]; });
    c.forEach(function (r, i) {
      var best = null, bestS = -Infinity;
      sounds.forEach(function (s) {
        var others = c.filter(function (o, j) { return j !== i && o.sound === s; });
        if (!others.length) return;
        var sc = dot(r.v, normalize(meanVec(others.map(function (o) { return o.v; }))));
        if (sc > bestS) { bestS = sc; best = s; }
      });
      res[r.sound][0] += best === r.sound ? 1 : 0; res[r.sound][1] += 1;
    });
    var rec = sounds.map(function (s) { return res[s][0] / res[s][1]; });
    return { perSound: res, balanced: rec.reduce(function (a, b) { return a + b; }, 0) / rec.length, chance: 1 / sounds.length };
  }

  root.CryEngine = { Model: Model, match: match, check: check, logMelPatches: logMelPatches, SR: SR, b64ToBytes: b64ToBytes };
})(typeof window !== "undefined" ? window : globalThis);
