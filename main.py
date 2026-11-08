from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    WORKBUDDY_CLIENT_ID: str
    WORKBUDDY_CLIENT_SECRET: str
    SUNO_API_KEY: str

    model_config = {'env_file': '.env', 'extra': 'ignore'}

settings = Settings()

import fastapi_cdn_host
import array
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import uuid
from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = Path(os.environ.get("SEPARATION_MEDIA_DIR", BASE_DIR / "media"))
MODEL_NAME = os.environ.get("DEMUCS_MODEL", "htdemucs_ft")
SHARED_TOKEN = os.environ.get("SEPARATION_TOKEN", "")
PUBLIC_BASE_URL = os.environ.get("SEPARATION_PUBLIC_URL", "").rstrip("/")
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", str(6 * 60 * 60)))
# 同时跑几个 Demucs：CPU 机器上并发跑多个会把内存/CPU 拖死，默认排队跑一个
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("MAX_CONCURRENT_JOBS", "1")))

MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="yinyu-separation", version="1.0")
fastapi_cdn_host.patch_docs(app)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

_jobs: Dict[str, dict] = {}
_lock = threading.Lock()
_slot = threading.Semaphore(MAX_CONCURRENT_JOBS)


class SubmitBody(BaseModel):
    songId: str = ""
    audioUrl: str
    title: str = ""
    model: Optional[str] = None
    stems: Optional[list] = None


def _auth(token: str) -> None:
    """共享密钥校验：留空表示不校验（仅限内网调试）。"""
    if SHARED_TOKEN and token != SHARED_TOKEN:
        raise HTTPException(status_code=401, detail="invalid separation token")


def _update(job_id: str, **patch) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.update(patch)
            job["updatedAt"] = time.time()


def _base_url() -> str:
    return PUBLIC_BASE_URL


def ffmpeg_bin() -> str:
    """找 ffmpeg：环境变量 > PATH > pip 安装的 imageio-ffmpeg 自带二进制。
    Windows 上很少有人手动装 ffmpeg，imageio-ffmpeg 是装 Python 依赖时最省事的一条路。"""
    env = os.environ.get("FFMPEG_BIN", "")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - 可选依赖，用到才导入
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return ""


def _clamp(value: float, low: float, high: float, fallback: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return fallback
    if num != num:  # NaN
        return fallback
    return max(low, min(high, num))


def _wav_rms(path: Path, seconds: float = 60.0) -> float:
    """估算一条 wav 的响度（0~1）。用来判断「伴奏是不是几乎是空的」——
    原唱本身是清唱录音时，Demucs 分离出的伴奏会接近静音，得如实告诉用户，
    不能让人对着一条没声音的伴奏唱。"""
    try:
        with wave.open(str(path), "rb") as fh:
            channels = fh.getnchannels()
            width = fh.getsampwidth()
            frames = fh.readframes(int(fh.getframerate() * seconds))
        if not frames or width != 2:
            return 0.0
        samples = array.array("h")
        samples.frombytes(frames)
        if channels > 1:
            samples = samples[::channels]
        if not samples:
            return 0.0
        total = sum(int(v) * int(v) for v in samples) / len(samples)
        return math.sqrt(total) / 32768.0
    except Exception:  # noqa: BLE001 - 估不出响度不该让整个任务失败
        return 0.0


def _download(url: str, target: Path) -> None:
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as res:
        res.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in res.iter_bytes():
                fh.write(chunk)


def _run_demucs(job_id: str, audio_url: str, model: str) -> None:
    """下载 → Demucs 两轨分离 → 转 mp3。整个过程把进度写回 job。"""
    with _slot:   # 排队：同时只跑 MAX_CONCURRENT_JOBS 个，其余在队列里等
        _run_demucs_locked(job_id, audio_url, model)


def _run_demucs_locked(job_id: str, audio_url: str, model: str) -> None:
    workdir = Path(tempfile.mkdtemp(prefix=f"sep_{job_id}_"))
    try:
        _update(job_id, status="processing", progress=6, message="正在下载原唱")
        source = workdir / "source.audio"
        _download(audio_url, source)

        _update(job_id, progress=18, message=f"正在用 {model} 分离音轨")
        # 官方 CLI：--two-stems=vocals 只拆出 vocals / no_vocals（=伴奏）
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-n", model,
            "-o", str(workdir / "out"),
            str(source),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "demucs failed")[-500:])

        _update(job_id, progress=78, message="正在导出伴奏 / 原唱")
        stem_dir = workdir / "out" / model / "source"
        vocals = stem_dir / "vocals.wav"
        accompaniment = stem_dir / "no_vocals.wav"
        if not accompaniment.exists():
            raise RuntimeError("没有生成伴奏轨")

        target_dir = MEDIA_DIR / job_id
        target_dir.mkdir(parents=True, exist_ok=True)
        accomp_out = target_dir / "accompaniment.mp3"
        vocals_out = target_dir / "vocals.mp3"
        # 320k 保证伴奏不会再被压糊；转不出来就退回 wav
        ffmpeg = ffmpeg_bin()
        if ffmpeg:
            subprocess.run([ffmpeg, "-y", "-i", str(accompaniment), "-b:a", "320k", str(accomp_out)],
                           capture_output=True, check=True)
            subprocess.run([ffmpeg, "-y", "-i", str(vocals), "-b:a", "320k", str(vocals_out)],
                           capture_output=True, check=True)
        else:
            accomp_out = stem_dir / "no_vocals.wav"
            vocals_out = stem_dir / "vocals.wav"
            shutil.copy(accomp_out, target_dir / accomp_out.name)
            shutil.copy(vocals_out, target_dir / vocals_out.name)
            accomp_out = target_dir / accomp_out.name
            vocals_out = target_dir / vocals_out.name

        base = _base_url()
        _update(
            job_id,
            status="done",
            progress=100,
            message="分离完成",
            accompanimentUrl=f"{base}/media/{job_id}/{accomp_out.name}",
            vocalsUrl=f"{base}/media/{job_id}/{vocals_out.name}",
            # 伴奏轨响度：接近 0 说明原唱本身几乎没有伴奏（清唱录音），客户端要如实提示
            accompanimentRms=round(_wav_rms(accompaniment), 5),
        )
    except Exception as error:  # noqa: BLE001 - 任务线程必须把失败写回 job
        _update(job_id, status="failed", progress=0, error=str(error)[:300])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _cleanup_loop() -> None:
    while True:
        time.sleep(600)
        cutoff = time.time() - JOB_TTL_SECONDS
        with _lock:
            for job_id in [k for k, v in _jobs.items() if v.get("updatedAt", 0) < cutoff]:
                _jobs.pop(job_id, None)
                shutil.rmtree(MEDIA_DIR / job_id, ignore_errors=True)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_cleanup_loop, daemon=True).start()


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model": MODEL_NAME, "ffmpeg": bool(ffmpeg_bin())}


@app.post("/jobs")
def create_job(body: SubmitBody, x_separation_token: Optional[str] = Header(default=None)) -> dict:
    _auth(x_separation_token or "")
    if not re.match(r"^https?://", body.audioUrl or ""):
        raise HTTPException(status_code=400, detail="audioUrl 必须是可下载的 http(s) 地址")
    # 同一首歌只跑一次：发布时后台已经提交过，之后任何人进 KTV 再提交都拿回同一个任务，
    # 这样「作者在发布时触发分离」和「唱的人轮询进度」才对得上，也避免重复烧 GPU。
    if body.songId:
        with _lock:
            for existing in _jobs.values():
                if existing.get("songId") == body.songId and existing.get("status") != "failed":
                    return {
                        "jobId": existing["jobId"],
                        "status": existing["status"],
                        "progress": existing.get("progress", 0),
                    }
    job_id = uuid.uuid4().hex[:16]
    with _lock:
        _jobs[job_id] = {
            "jobId": job_id,
            "songId": body.songId,
            "title": body.title,
            "status": "queued",
            "progress": 0,
            "accompanimentUrl": "",
            "vocalsUrl": "",
            "error": "",
            "createdAt": time.time(),
            "updatedAt": time.time(),
        }
    threading.Thread(
        target=_run_demucs,
        args=(job_id, body.audioUrl, body.model or MODEL_NAME),
        daemon=True,
    ).start()
    return {"jobId": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def read_job(job_id: str, x_separation_token: Optional[str] = Header(default=None)) -> dict:
    _auth(x_separation_token or "")
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "jobId": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "accompanimentUrl": job["accompanimentUrl"],
        "vocalsUrl": job["vocalsUrl"],
        "error": job["error"],
        "message": job.get("message", ""),
        "accompanimentRms": job.get("accompanimentRms"),
    }


@app.post("/mix")
def mix_track(
    file: UploadFile = File(...),
    accompanimentUrl: str = Form(""),
    vocalOffset: float = Form(0.0),
    vocalGain: float = Form(1.0),
    accompanimentGain: float = Form(0.8),
    x_separation_token: Optional[str] = Header(default=None),
) -> dict:
    """人声 + 伴奏 → m4a。小程序端优先用 wx.createMediaContainer 本地合成，
    这里是本地不支持时的兜底（ffmpeg amix），产物同样是 audio-only mp4(m4a)。"""
    _auth(x_separation_token or "")
    if not re.match(r"^https?://", accompanimentUrl or ""):
        raise HTTPException(status_code=400, detail="accompanimentUrl 必须是可下载的 http(s) 地址")
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        raise HTTPException(status_code=503, detail="服务器没有安装 ffmpeg，无法混流")

    job_id = uuid.uuid4().hex[:16]
    workdir = Path(tempfile.mkdtemp(prefix=f"mix_{job_id}_"))
    try:
        suffix = Path(file.filename or "").suffix or ".wav"
        vocal = workdir / ("vocal" + suffix)
        with open(vocal, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        if vocal.stat().st_size < 1024:
            raise HTTPException(status_code=400, detail="上传的人声文件是空的")
        accomp = workdir / "accompaniment.audio"
        _download(accompanimentUrl, accomp)

        out_dir = MEDIA_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "mix.m4a"
        offset = _clamp(vocalOffset, 0.0, 30.0, 0.0)
        # 录音机比伴奏早起嘴的那段要剪掉，混完才是同一条时间轴
        trim = ["-ss", f"{offset:.3f}"] if offset > 0.02 else []
        vgain = _clamp(vocalGain, 0.0, 2.0, 1.0)
        again = _clamp(accompanimentGain, 0.0, 1.0, 0.8)
        base_filter = (
            f"[0:a]volume={vgain:.2f}[v];"
            f"[1:a]volume={again:.2f}[a];"
            "[v][a]amix=inputs=2:duration=longest:dropout_transition=0%s[out]"
        )
        attempts = [base_filter % ":normalize=0", base_filter % ""]
        last_error = ""
        for filter_complex in attempts:
            cmd = ([ffmpeg, "-y"] + trim + ["-i", str(vocal), "-i", str(accomp)]
                   + ["-filter_complex", filter_complex, "-map", "[out]",
                      "-c:a", "aac", "-b:a", "192k", "-f", "mp4", str(out)])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and out.exists():
                return {
                    "url": f"{_base_url()}/media/{job_id}/{out.name}",
                    "engine": "ffmpeg",
                    "duration": None,
                }
            # 老版本 ffmpeg 不认 normalize 参数：去掉再试一次
            last_error = (proc.stderr or "")[-400:]
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"ffmpeg 混流失败：{last_error}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# 混响产物保留 3 天（跟唱时用一次就够，不需要长期存）
FX_MEDIA_TTL = float(os.environ.get("FX_MEDIA_TTL", 3 * 86400))

# ============================================================
# 人声房间混响（跟唱页的「同一空间感」）
#
# 小程序 InnerAudioContext 没有效果器，实时混响做不了，
# 所以在服务端用 ffmpeg 预处理一份带混响的人声，前端只把播放源换过去。
#   POST /vocals/reverb   multipart: file / wet / duration
#   → {"url": "...", "wet": 0.15, "engine": "ffmpeg"}
# 时长按 duration 截断，保证换了人声之后双轨的时间轴不变。
# ============================================================

@app.post("/vocals/reverb")
def vocals_reverb(
    file: UploadFile = File(...),
    wet: float = Form(0.15),
    duration: float = Form(0.0),
    x_separation_token: Optional[str] = Header(default=None),
) -> dict:
    """给干声加一点房间混响（干湿比默认 15%），返回可直接播放的固定地址。"""
    _auth(x_separation_token or "")
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        raise HTTPException(status_code=503, detail="服务器没有安装 ffmpeg，无法加混响")

    ratio = _clamp(wet, 0.0, 0.5, 0.15)
    job_id = uuid.uuid4().hex[:16]
    workdir = Path(tempfile.mkdtemp(prefix=f"reverb_{job_id}_"))
    try:
        suffix = Path(file.filename or "").suffix or ".mp3"
        vocal = workdir / ("vocal" + suffix)
        with open(vocal, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        if vocal.stat().st_size < 1024:
            raise HTTPException(status_code=400, detail="上传的人声文件是空的")

        out_dir = MEDIA_DIR / "fx" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "vocal-room.mp3"

        # 湿声：aecho 造两组早期反射（27ms / 53ms）+ 一点尾巴，听感像小房间；
        # 干声按 (1-wet) 衰减，湿声按 wet / 0.8 补回 aecho 的 out_gain 损失，
        # 这样实际干湿比才真的接近 15%，不会一开混响人声就变小。
        dry_gain = max(0.0, 1.0 - ratio)
        wet_gain = ratio / 0.8
        filter_complex = (
            "[0:a]asplit=2[dry][src];"
            "[src]aecho=0.8:0.6:27|53:0.22|0.14[wet];"
            f"[dry]volume={dry_gain:.3f}[d];[wet]volume={wet_gain:.3f}[w];"
            "[d][w]amix=inputs=2:duration=first:normalize=0[out]"
        )
        cmd = [ffmpeg, "-y", "-i", str(vocal),
               "-filter_complex", filter_complex, "-map", "[out]",
               "-c:a", "libmp3lame", "-b:a", "192k"]
        limit = _clamp(duration, 0.0, 900.0, 0.0)
        if limit > 0.5:
            cmd += ["-t", f"{limit:.3f}"]      # 截掉混响尾巴，别让时长变长
        cmd += [str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
            raise HTTPException(status_code=500,
                                detail="ffmpeg 混响处理失败：" + (proc.stderr or "")[-400:])
        return {
            "url": f"{_base_url()}/media/fx/{job_id}/{out.name}",
            "wet": round(ratio, 2),
            "engine": "ffmpeg",
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ============================================================
# Suno 伴奏生成（「录音 + AI 配伴奏」的服务端）
#
# 小程序端只说「要什么风格、多长」，Suno 的 API key 永远不出现在前端：
#   1. 小程序 POST /suno/backing { stylePrompt, duration, title }
#   2. 服务端用 SUNO_API_KEY 调 Suno 中转，instrumental=true（纯器乐、无人声）
#   3. 生成完把音频下载到本地 media，返回稳定 URL（第三方 CDN 链接会过期）
#   4. 小程序轮询 GET /suno/backing/{jobId} 拿状态与地址
#
# 环境变量：
#   SUNO_API_BASE   Suno 中转地址（默认 https://api.acedata.cloud，Ace Data Cloud）
#   SUNO_API_KEY    必须，服务端持有；留空时接口返回 503 并说明未配置
#   SUNO_MODEL      模型名（默认 chirp-v5-5，可选 chirp-v6 / chirp-v6-mini / chirp-v4-5）
#   SUNO_PROVIDER   acedata（默认）| mock（本机占位伴奏，仅供开发联调，会标注 engine=mock）
#   SUNO_TIMEOUT    单个任务最长等待秒数（默认 420）
#
# Ace Data Cloud 的协议（和 sunoapi.org 完全不同）：
#   提交  POST /suno/audios   {action:"generate", prompt, model, instrumental:true, async:true}
#         → {"task_id": "..."}          # async=true 立刻返回，不阻塞
#   查询  POST /suno/tasks    {action:"retrieve", id: task_id}
#         → {response:{success, data:[{state, audio_url, duration, ...}]}, finished_at}
#   state: pending | running | succeeded | error
# ============================================================
SUNO_API_BASE = os.environ.get("SUNO_API_BASE", "https://api.acedata.cloud").rstrip("/")
SUNO_API_KEY = os.environ.get("SUNO_API_KEY", "")
SUNO_MODEL = os.environ.get("SUNO_MODEL", "chirp-v5-5")
SUNO_PROVIDER = (os.environ.get("SUNO_PROVIDER", "acedata") or "acedata").lower()
SUNO_TIMEOUT = int(os.environ.get("SUNO_TIMEOUT", "420"))
SUNO_MEDIA_TTL = int(os.environ.get("SUNO_MEDIA_TTL", str(14 * 24 * 3600)))
SUNO_MEDIA_DIR = Path(os.environ.get("SUNO_MEDIA_DIR", BASE_DIR / "media" / "suno"))
SUNO_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

SUNO_MAX_SECONDS = 300.0   # 人声时长上限，超了按 300 秒提交

_suno_jobs: Dict[str, dict] = {}
_suno_lock = threading.Lock()


class SunoBody(BaseModel):
    stylePrompt: str = ""
    duration: float = 0
    title: str = ""
    vocalsUrl: str = ""


def _suno_update(job_id: str, **patch) -> None:
    with _suno_lock:
        job = _suno_jobs.get(job_id)
        if job:
            job.update(patch)
            job["updatedAt"] = time.time()


def _suno_snapshot(job_id: str) -> Optional[dict]:
    with _suno_lock:
        job = _suno_jobs.get(job_id)
        return dict(job) if job else None


def _suno_base_url() -> str:
    """本地调试没配公网地址时返回相对路径，小程序端会自己补服务端前缀。"""
    return PUBLIC_BASE_URL


def _suno_headers() -> dict:
    return {
        "Authorization": f"Bearer {SUNO_API_KEY}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }


def _suno_supports_duration(model: str) -> bool:
    """duration 只在新模型上有效；老模型（v3-5 / v4）传了会 400，直接不发。"""
    name = (model or "").lower()
    return any(tag in name for tag in ("v5-5", "v6", "v6-mini", "v6-wild", "v5"))


def _suno_error_text(prefix: str, status: int, body: dict) -> str:
    """把 Ace Data 的错误翻成人话。用户看到的不能是一坨 JSON。"""
    err = body.get("error") if isinstance(body, dict) else None
    code = ""
    detail = ""
    if isinstance(err, dict):
        code = str(err.get("code") or "")
        detail = str(err.get("message") or "")
    elif err:
        detail = str(err)
    friendly = {
        "used_up": "Suno 额度不足：Ace Data Cloud 账户余额已用完，充值后就能继续生成",
        "insufficient_balance": "Suno 额度不足：账户余额不够生成一首，请先充值",
        "unauthorized": "SUNO_API_KEY 无效或已过期，请在环境变量里换成新的 key",
        "invalid_token": "SUNO_API_KEY 无效或已过期，请检查环境变量",
        "rate_limit": "Suno 调用太频繁，等十几秒再试",
        "model_unavailable": f"模型 {SUNO_MODEL} 当前不可用，换个 SUNO_MODEL 试试",
    }.get(code)
    if not friendly and status == 401:
        friendly = "SUNO_API_KEY 无效或已过期，请在环境变量里换成新的 key"
    if not friendly and status == 429:
        friendly = "Suno 调用太频繁，等十几秒再试"
    if friendly:
        return f"{friendly}（{code or status}）"
    return f"{prefix} HTTP {status}：{detail or str(body)[:200]}"


def _suno_net_text(prefix: str) -> str:
    return f"{prefix}：连不上 Suno 服务（网络或代理波动），请稍后重试"


def _suno_submit(prompt: str, duration: float, title: str) -> str:
    """提交生成任务，立刻拿回 task_id（async=true，不等渲染）。

    Ace Data Cloud：POST /suno/audios
      instrumental=true → 纯器乐，lyric 会返回 "[Instrumental]"，不会有人声
      async=true        → 不阻塞、也不需要 callback_url，交给 /suno/tasks 轮询
    """
    payload = {
        "action": "generate",
        "prompt": prompt,
        "model": SUNO_MODEL,
        "instrumental": True,
        "async": True,
    }
    if duration and _suno_supports_duration(SUNO_MODEL):
        # 让伴奏长度贴着人声走；平台侧是「倾向性提示」，不是硬裁剪
        payload["duration"] = int(round(max(10.0, min(360.0, float(duration)))))
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        try:
            res = client.post(f"{SUNO_API_BASE}/suno/audios", json=payload,
                              headers=_suno_headers())
        except httpx.HTTPError:
            # 提交是有副作用（计费）的，网络断在半路不重试，避免重复生成一首
            raise RuntimeError(_suno_net_text("Suno 提交失败"))
        body = res.json() if res.content else {}
        if res.status_code >= 400 or body.get("success") is False:
            raise RuntimeError(_suno_error_text("Suno 提交失败", res.status_code, body))
        data = body.get("data")
        task_id = str(body.get("task_id") or body.get("taskId") or "")
        if not task_id and isinstance(data, dict):
            task_id = str(data.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"Suno 没有返回任务号：{str(body)[:300]}")
        return task_id


def _suno_pick(item: dict) -> dict:
    """一条生成结果 → 统一结构。字段名在 Ace Data 上是 snake_case。"""
    state = str(item.get("state") or "").lower()
    audio_url = str(item.get("audio_url") or item.get("audioUrl") or "")
    error = str(item.get("error") or item.get("error_message") or "")
    try:
        duration = float(item.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    return {"state": state, "audioUrl": audio_url, "duration": duration, "error": error}


def _suno_fetch(task_id: str) -> dict:
    """查任务状态。Ace Data Cloud：POST /suno/tasks {action:"retrieve", id}。

    任务还没跑出 response 时接口返回 {}，这时按「排队中」处理，不当作失败。
    """
    # 查询是只读的，网络抖一下重试三次，别让一次 SSL EOF 把整个任务判死
    for attempt in range(3):
        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                res = client.post(f"{SUNO_API_BASE}/suno/tasks",
                                  json={"action": "retrieve", "id": task_id},
                                  headers=_suno_headers())
                body = res.json() if res.content else {}
            break
        except httpx.HTTPError:
            if attempt < 2:
                time.sleep(2)
    else:
        raise RuntimeError(_suno_net_text("Suno 查询失败"))
    if res.status_code >= 400:
        raise RuntimeError(_suno_error_text("Suno 查询失败", res.status_code, body))
    response = body.get("response") or {}
    items = response.get("data") or body.get("data") or []
    item = {}
    if isinstance(items, list) and items:
        item = items[0] if isinstance(items[0], dict) else {}
    elif isinstance(items, dict):
        item = items
    picked = _suno_pick(item)
    if not picked["state"]:
        picked["state"] = "pending" if not body.get("finished_at") else "succeeded"
    if response.get("success") is False:
        picked["state"] = "failed"
        picked["error"] = picked["error"] or str(response.get("error") or "Suno 生成失败")
    return picked


def _suno_progress(elapsed: float, state: str) -> tuple:
    """Ace Data 只给 pending / running / succeeded，没有细阶段；
    按已等时长推一个平滑进度，别让用户盯着 18% 干等。"""
    if state == "succeeded":
        return ("rendering", 85, "伴奏渲染完成，正在取回音频")
    if state == "running":
        span = max(0.0, min(1.0, elapsed / 75.0))   # 一般 30~90 秒出结果
        return ("rendering", int(30 + 45 * span), "AI 正在写伴奏…（编曲渲染中）")
    return ("writing", 18, "AI 正在写伴奏…（排队中）")


def _mock_backing(job_id: str, duration: float) -> Path:
    """开发占位伴奏：五声音阶拨弦 + 轻雨噪。
    只在 SUNO_PROVIDER=mock 时使用，返回 engine=mock，前端会明确标注
    「开发占位伴奏，不是 Suno 生成」，绝不当成真结果。"""
    seconds = max(8.0, min(180.0, float(duration or 30)))
    rate = 22050
    total = int(rate * seconds)
    samples = [0.0] * total
    # D 宫五声音阶，两个八度多一点：听起来就是古风拨弦
    scale = [293.66, 329.63, 369.99, 440.00, 493.88,
             587.33, 659.25, 739.99, 880.00, 987.77]
    rnd = random.Random(abs(hash(job_id)) % (2 ** 31))
    t = 0.0
    while t < seconds:
        freq = scale[rnd.randrange(len(scale))]
        start = int(t * rate)
        length = min(int(rate * 1.8), total - start)
        for i in range(length):
            env = math.exp(-i / (rate * 0.60))
            samples[start + i] += 0.26 * math.sin(2 * math.pi * freq * i / rate) * env
        t += 0.42 + 0.5 * rnd.random()
    for i in range(total):  # 一层很轻的雨
        samples[i] += 0.010 * (rnd.random() * 2 - 1)
    peak = max(1e-6, max(abs(v) for v in samples))
    gain = 0.86 / peak
    frames = array.array("h")
    for v in samples:
        frames.append(int(max(-1.0, min(1.0, v * gain)) * 32000))
    out_dir = SUNO_MEDIA_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "backing.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(frames.tobytes())
    return path


def _suno_run(job_id: str, prompt: str, duration: float, title: str) -> None:
    """提交 → 轮询 → 下载到本地 → 完成。全程把阶段写回 job。"""
    try:
        if SUNO_PROVIDER == "mock":
            _suno_update(job_id, status="processing", stage="writing", progress=25,
                         message="AI 正在写伴奏…")
            time.sleep(1.2)
            _suno_update(job_id, stage="rendering", progress=70, message="AI 正在写伴奏…（渲染中）")
            path_done = _mock_backing(job_id, duration)
            _suno_update(job_id, stage="downloading", progress=92, message="正在保存伴奏")
            name = path_done.name
            _suno_update(
                job_id, status="done", stage="done", progress=100,
                message="伴奏已生成（开发占位版）",
                accompanimentUrl=f"{_suno_base_url()}/media/suno/{job_id}/{name}",
                engine="mock",
                audioDuration=duration,
            )
            return

        _suno_update(job_id, stage="writing", progress=12, message="AI 正在写伴奏…")
        started = time.time()
        task_id = _suno_submit(prompt, duration, title)
        _suno_update(job_id, status="processing", taskId=task_id, progress=18)

        deadline = started + SUNO_TIMEOUT
        audio_url = ""
        audio_duration = duration
        while time.time() < deadline:
            time.sleep(4)   # Ace Data 一般 30~90 秒出结果，4 秒问一次够了
            state = _suno_fetch(task_id)
            raw = (state["state"] or "").lower()
            if raw in ("failed", "error"):
                raise RuntimeError(state["error"] or "Suno 生成失败（可能触发了内容审核）")
            stage, progress, message = _suno_progress(time.time() - started, raw)
            _suno_update(job_id, status="processing", stage=stage, progress=progress,
                         message=message, sunoStatus=raw)
            if state["audioUrl"]:
                audio_url = state["audioUrl"]
                if state["duration"]:
                    audio_duration = state["duration"]
                break
        if not audio_url:
            raise RuntimeError("Suno 生成超时：可以稍后重试，或换一种风格描述")

        _suno_update(job_id, stage="downloading", progress=90, message="正在取回伴奏")
        out_dir = SUNO_MEDIA_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "backing.mp3"
        _download(audio_url, target)
        if not target.exists() or target.stat().st_size < 2048:
            raise RuntimeError("拿到的伴奏文件是空的，请重试")
        _suno_update(
            job_id, status="done", stage="done", progress=100, message="伴奏已生成",
            accompanimentUrl=f"{_suno_base_url()}/media/suno/{job_id}/{target.name}",
            engine="suno", audioDuration=audio_duration,
        )
    except Exception as exc:  # noqa: BLE001 - 任何失败都要如实回到前端
        _suno_update(job_id, status="failed", stage="failed", progress=0,
                     message=str(exc)[:400], error=str(exc)[:400])


def _suno_cleanup_loop() -> None:
    while True:
        time.sleep(3600)
        cutoff = time.time() - SUNO_MEDIA_TTL
        try:
            for path in SUNO_MEDIA_DIR.iterdir():
                if path.is_dir() and path.stat().st_mtime < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass
        # 混响产物只是中转文件，跟唱时用完就不再需要，3 天后清掉
        try:
            fx_root = MEDIA_DIR / "fx"
            fx_cutoff = time.time() - FX_MEDIA_TTL
            if fx_root.exists():
                for path in fx_root.iterdir():
                    if path.is_dir() and path.stat().st_mtime < fx_cutoff:
                        shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


@app.on_event("startup")
def _suno_startup() -> None:
    threading.Thread(target=_suno_cleanup_loop, daemon=True).start()


@app.get("/suno/healthz")
def suno_healthz() -> dict:
    return {
        "ok": bool(SUNO_PROVIDER == "mock" or SUNO_API_KEY),
        "provider": SUNO_PROVIDER,
        "base": SUNO_API_BASE,
        "model": SUNO_MODEL,
        "configured": bool(SUNO_API_KEY) or SUNO_PROVIDER == "mock",
    }


@app.post("/suno/backing")
def suno_create(body: SunoBody, x_separation_token: Optional[str] = Header(default=None)) -> dict:
    _auth(x_separation_token or "")
    prompt = (body.stylePrompt or "").strip()
    if len(prompt) < 4:
        raise HTTPException(status_code=400, detail="风格描述太短，至少写 4 个字")
    if SUNO_PROVIDER != "mock" and not SUNO_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="服务端还没配置 SUNO_API_KEY：在服务器环境变量里填上 Suno 中转的 key 才能生成伴奏",
        )
    duration = _clamp(body.duration, 8.0, SUNO_MAX_SECONDS, 30.0)
    job_id = uuid.uuid4().hex[:16]
    with _suno_lock:
        _suno_jobs[job_id] = {
            "jobId": job_id,
            "title": (body.title or "").strip()[:40],
            "stylePrompt": prompt[:200],
            "requestedDuration": duration,
            "status": "queued",
            "stage": "queued",
            "progress": 5,
            "message": "AI 正在写伴奏…（排队中）",
            "accompanimentUrl": "",
            "engine": "mock" if SUNO_PROVIDER == "mock" else "suno",
            "audioDuration": 0,
            "error": "",
            "createdAt": time.time(),
            "updatedAt": time.time(),
        }
    threading.Thread(target=_suno_run,
                     args=(job_id, prompt, duration, body.title or ""),
                     daemon=True).start()
    return {"jobId": job_id, "status": "queued", "progress": 5}


@app.get("/suno/backing/{job_id}")
def suno_read(job_id: str, x_separation_token: Optional[str] = Header(default=None)) -> dict:
    _auth(x_separation_token or "")
    job = _suno_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="backing job not found")
    return {
        "jobId": job_id,
        "status": job["status"],
        "stage": job.get("stage", ""),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "accompanimentUrl": job.get("accompanimentUrl", ""),
        "engine": job.get("engine", ""),
        "duration": job.get("audioDuration", 0),
        "error": job.get("error", ""),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)