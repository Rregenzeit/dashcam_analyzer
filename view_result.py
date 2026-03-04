"""
CLI 결과 뷰어 - dashcam_analyzer 분석 결과를 터미널에서 확인

사용법:
  python view_result.py                    # 최근 job 결과 표시
  python view_result.py JOB_ID            # 특정 job 결과 표시
  python view_result.py JOB_ID --open-clips  # 결과 + 클립 재생
  python view_result.py --list            # 모든 job 목록
"""
import sys
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from config import OUTPUTS_DIR
except ImportError:
    OUTPUTS_DIR = "outputs"


def list_jobs():
    outputs = Path(OUTPUTS_DIR)
    if not outputs.exists():
        print("아직 분석된 영상이 없습니다.")
        return

    jobs = []
    for job_dir in sorted(outputs.iterdir()):
        job_file = job_dir / "job.json"
        if job_file.exists():
            try:
                info = json.loads(job_file.read_text(encoding="utf-8"))
                jobs.append(info)
            except Exception:
                pass

    if not jobs:
        print("저장된 job이 없습니다.")
        return

    print(f"\n{'Job ID':<38} {'상태':<12} {'진행률':>6}  생성 시각")
    print("-" * 80)
    for j in jobs:
        created = datetime.fromtimestamp(j.get("created_at", 0)).strftime("%Y-%m-%d %H:%M:%S")
        progress = f"{j.get('progress', 0)*100:.0f}%"
        print(f"{j['job_id']:<38} {j['status']:<12} {progress:>6}  {created}")
    print()


def get_latest_job_id():
    outputs = Path(OUTPUTS_DIR)
    if not outputs.exists():
        return None

    completed = []
    for job_dir in outputs.iterdir():
        result_file = job_dir / "result.json"
        if result_file.exists():
            try:
                r = json.loads(result_file.read_text(encoding="utf-8"))
                completed.append((r.get("finished_at", 0), job_dir.name))
            except Exception:
                pass

    if not completed:
        return None
    return sorted(completed, reverse=True)[0][1]


def open_clip(clip_path: Path):
    if not clip_path.exists():
        print(f"  [오류] 클립 없음: {clip_path}")
        return
    try:
        if sys.platform == "win32":
            os.startfile(str(clip_path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(clip_path)])
        else:
            subprocess.run(["xdg-open", str(clip_path)])
        print(f"  재생 중: {clip_path.name}")
    except Exception as e:
        print(f"  [오류] 재생 실패: {e}")


def show_result(job_id: str, open_clips: bool = False):
    result_path = Path(OUTPUTS_DIR) / job_id / "result.json"
    job_path = Path(OUTPUTS_DIR) / job_id / "job.json"

    if not result_path.exists():
        if job_path.exists():
            info = json.loads(job_path.read_text(encoding="utf-8"))
            print(f"\nJob ID : {job_id}")
            print(f"상태   : {info['status']}")
            print(f"진행률 : {info.get('progress', 0)*100:.0f}%")
            print(f"메시지 : {info.get('message', '')}")
        else:
            print(f"[오류] job을 찾을 수 없습니다: {job_id}")
        return

    result = json.loads(result_path.read_text(encoding="utf-8"))
    events = result.get("events", [])
    finished = result.get("finished_at")
    finished_str = datetime.fromtimestamp(finished).strftime("%Y-%m-%d %H:%M:%S") if finished else "N/A"

    print("\n" + "=" * 70)
    print(f"  Dashcam Cut-In 분석 결과")
    print("=" * 70)
    print(f"  Job ID    : {job_id}")
    print(f"  완료 시각 : {finished_str}")
    print(f"  끼어들기  : {len(events)}건")
    print("=" * 70)

    if not events:
        print("\n  감지된 끼어들기 이벤트 없음\n")
        return

    print(f"\n  {'#':<4} {'이벤트 ID':<14} {'시작F':>6} {'종료F':>6} {'트랙':>5}  {'번호판':<12} {'방향등':<8} 클립")
    print("  " + "-" * 68)

    clips_dir = Path(OUTPUTS_DIR) / job_id / "clips"

    for i, evt in enumerate(events, 1):
        plate = evt.get("plate_text") or "-"
        signal = evt.get("turn_signal", "unknown")
        signal_icon = "ON" if signal == "on" else ("OFF" if signal == "off" else "?")
        clip_name = evt.get("clip_filename", "")
        clip_exists = "O" if (clips_dir / clip_name).exists() else "X"

        print(f"  {i:<4} {evt['event_id']:<14} {evt['frame_start']:>6} {evt['frame_end']:>6} "
              f"{evt['track_id']:>5}  {plate:<12} {signal_icon:<8} {clip_exists} {clip_name}")

        if open_clips and clip_name:
            open_clip(clips_dir / clip_name)

    print()
    if not open_clips and events:
        print(f"  클립 경로: {clips_dir}")
        print(f"  클립 재생: python view_result.py {job_id} --open-clips")
    print()


def main():
    args = sys.argv[1:]

    if "--list" in args:
        list_jobs()
        return

    open_clips = "--open-clips" in args
    args = [a for a in args if not a.startswith("--")]

    if args:
        job_id = args[0]
    else:
        job_id = get_latest_job_id()
        if not job_id:
            print("완료된 분석 결과가 없습니다. --list 로 전체 목록을 확인하세요.")
            return
        print(f"(최근 job: {job_id})")

    show_result(job_id, open_clips=open_clips)


if __name__ == "__main__":
    main()
