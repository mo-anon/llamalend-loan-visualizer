"""Re-render a video from cached block_data.json — no RPC calls needed.

Usage:
  uv run python rerender.py output/2026-03-26_CRV-crvUSD_0x9F2F/block_data.json
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import font_manager, rcParams

from visualize_loan import (
    CHART_COLORS,
    EVENT_LABELS,
    EVENT_ICONS,
    EVENT_STYLE,
    _fill_band_column,
    _fmt_event_line,
    _progress_bar,
    get_event_style,
    get_health_color,
)

BASE_DIR = Path(__file__).resolve().parent
font_path = BASE_DIR / "MonaSans-Regular.ttf"
font_manager.fontManager.addfont(str(font_path))
monasans_font = font_manager.FontProperties(fname=str(font_path))
rcParams['font.sans-serif'] = [monasans_font.get_name()]


def rerender(block_data_path: str, log_fn=print) -> Path:
    """Load block_data.json and run only the render + encode passes."""

    block_data_path = Path(block_data_path)
    if block_data_path.is_dir():
        block_data_path = block_data_path / "block_data.json"
    if not block_data_path.exists():
        raise FileNotFoundError(f"Not found: {block_data_path}")

    log_fn(f"Loading cached data from {block_data_path}")
    with block_data_path.open() as f:
        cached = json.load(f)

    all_block_data = cached["blocks"]
    events = cached.get("events", [])

    # Relabel loan creation events (first borrow, or borrow after close/liquidation)
    for i, event in enumerate(events):
        if event.get("action") != "borrow":
            continue
        if i == 0 or events[i - 1].get("action") in ("closed", "liquidated"):
            event["action"] = "loan_created"

    market_label = cached["market"]
    base_symbol = cached["base_symbol"]
    collateral_symbol = cached["collateral_symbol"]

    log_fn(f"  Market: {market_label}")
    log_fn(f"  Blocks: {len(all_block_data)}")
    log_fn(f"  Events: {len(events)}")

    # --- Output directory (sibling of the original) -------------------------
    run_dir = block_data_path.parent
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    file_str = f"{base_symbol}_{collateral_symbol}"
    mp4_filename = run_dir / "video.mp4"

    # --- Compute axis ranges ------------------------------------------------
    all_prices = []
    all_band_nums = set()
    all_health_values = []
    for bd in all_block_data:
        all_prices.append(bd['price_oracle'])
        all_health_values.append(bd['health'])
        for band in bd['bands']:
            all_prices.append(band['min_price'])
            all_prices.append(band['max_price'])
            all_band_nums.add(band['band_num'])

    if all_prices:
        price_min = min(all_prices)
        price_max = max(all_prices)
        price_range = price_max - price_min
        price_center = (price_min + price_max) / 2
        expanded_range = price_range * 2.0
        y_min = price_center - expanded_range / 2
        y_max = price_center + expanded_range / 2
    else:
        y_min, y_max = 2700, 3200

    if all_health_values:
        health_max = max(all_health_values)
        health_min = min(all_health_values)
        health_y_max = (health_max * 1.1 if health_max > 0
                        else max(0, health_max * 0.9) + 1)
        health_y_min = (health_min * 1.1 if health_min < 0
                        else health_min * 0.9 if health_min > 0
                        else -1)
        health_y_max = int(health_y_max) + (1 if health_y_max > 0 else 0)
        health_y_min = int(health_y_min) - (1 if health_y_min < 0 else 0)
    else:
        health_y_max, health_y_min = 5, -1

    # --- Map events to nearest block index ----------------------------------
    events_by_block_index = {}
    close_loan_block_index = None

    if events:
        fetched_blocks = [bd['block'] for bd in all_block_data]
        health_values = [bd['health'] for bd in all_block_data]

        for event in events:
            event_block = event['blocknumber']
            is_close_event = event.get('action') in ['closed', 'liquidated']

            closest_idx = min(range(len(fetched_blocks)),
                              key=lambda i: abs(fetched_blocks[i] - event_block))

            if is_close_event:
                close_loan_block_index = closest_idx

            health_after = health_values[closest_idx] if closest_idx < len(health_values) else None
            health_before = health_values[closest_idx - 1] if closest_idx > 0 else None
            health_increase = (health_after - health_before
                               if health_after is not None and health_before is not None
                               else None)

            events_by_block_index.setdefault(closest_idx, []).append({
                'event': event,
                'health_increase': health_increase,
                'health_before': health_before,
                'health_after': health_after,
            })

    # --- Render frames ------------------------------------------------------
    log_fn("\n" + "=" * 60)
    log_fn("Rendering frames from cached data...")
    log_fn("=" * 60)

    price_history = []
    time_indices = []
    image_start_time = time.time()

    dpi_num = 150
    fig = plt.figure(figsize=(10.5, 5.6), dpi=dpi_num)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.3], width_ratios=[6, 1.4],
                          hspace=0.25, wspace=0.25)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_health = fig.add_subplot(gs[0, 1])
    ax_events = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_events.axis('off')
    ax_stats.axis('off')

    ax_main.grid(axis='y', linestyle='--', alpha=0.5)
    ax_main.grid(axis='x', linestyle='--', alpha=0.3)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlim(0, len(all_block_data))
    ax_main.set_xlabel("Time (Block Steps)")
    ax_main.set_ylabel(f"{collateral_symbol} Price ({base_symbol})")
    ax_main.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.2f}'))

    price_line, = ax_main.plot([], [], color="#3162F4", linestyle='-', linewidth=2.5,
                               marker='o', markersize=2, alpha=0.8)

    base_legend_handles = [
        Line2D([], [], color="#3162F4", linewidth=2.5, marker='o', markersize=2, alpha=0.8),
        Patch(facecolor=CHART_COLORS['collateral']['fill'], edgecolor=CHART_COLORS['collateral']['edge'], alpha=0.6),
        Patch(facecolor=CHART_COLORS['crvusd']['fill'], edgecolor=CHART_COLORS['crvusd']['edge'], alpha=0.6),
    ]
    base_legend_labels = ["Oracle Price", f"Collateral {collateral_symbol}", f"Collateral {base_symbol}"]
    legend_seen_actions = set()
    event_legend_handles = []
    event_legend_labels = []
    ax_main.legend(base_legend_handles, base_legend_labels, loc='upper left')

    ax_health.set_xlim(-0.8, 0.8)
    ax_health.set_xticks([])
    ax_health.set_ylim(health_y_min, health_y_max)
    ax_health.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    ax_health.set_ylabel("Health")

    health_bar_artist = ax_health.bar(0, 0, width=0.6, color="#4CAF50",
                                      edgecolor="#2E7D32", linewidth=0.8)[0]
    health_text_artist = ax_health.text(0, 0, '', ha='center', va='bottom',
                                        fontsize=9, fontweight='bold', color="#2E7D32")
    stats_text_artist = ax_stats.text(0, 1, '', transform=ax_stats.transAxes,
                                      fontsize=8, verticalalignment='top',
                                      horizontalalignment='left',
                                      family='monospace', color="#1d1d1f")
    events_text_artist = ax_events.text(0, 1, '', transform=ax_events.transAxes,
                                        fontsize=7.3, verticalalignment='top',
                                        horizontalalignment='left',
                                        family='monospace', color="#1d1d1f")

    fig.tight_layout(rect=[0.03, 0.02, 0.98, 0.97])

    bands_pixel_h = 600
    bands_image = np.zeros((bands_pixel_h, len(all_block_data), 4), dtype=np.float32)
    bands_im = ax_main.imshow(
        bands_image, extent=[0, len(all_block_data), y_min, y_max],
        aspect='auto', origin='upper', interpolation='nearest', zorder=0,
    )

    plotted_event_indices = set()
    max_band_idx = close_loan_block_index if close_loan_block_index is not None else len(all_block_data)

    total_images = len(all_block_data)
    render_interval = max(1, total_images // 50)

    for idx, block_data in enumerate(all_block_data):
        cycle = idx + 1
        block = block_data['block']

        if cycle == 1 or cycle == total_images or cycle % render_interval == 0:
            if cycle == 1:
                log_fn(f"{_progress_bar(0, total_images)} | Rendering frame 1/{total_images} (block {block})")
            else:
                elapsed_time = time.time() - image_start_time
                eta = (elapsed_time / (cycle - 1)) * (total_images - cycle + 1)
                log_fn(f"{_progress_bar(cycle, total_images)} | Rendering frame {cycle}/{total_images} (block {block}) | ETA: {eta:.1f}s")

        price_oracle = block_data['price_oracle']
        health = block_data['health']
        block_date = block_data['block_date']
        bands = block_data['bands']

        price_history.append(price_oracle)
        time_indices.append(idx)

        if idx < max_band_idx:
            _fill_band_column(bands_image, idx, bands, y_min, y_max, CHART_COLORS)
            bands_im.set_data(bands_image)

        price_line.set_data(time_indices, price_history)

        ax_main.set_title(
            f"{market_label} Mint Market - Time Series, Block: {block}, Time: {block_date}",
            pad=12)

        if events:
            for event_idx, event_list in events_by_block_index.items():
                if event_idx > idx or event_idx in plotted_event_indices:
                    continue
                plotted_event_indices.add(event_idx)
                for event_data in event_list:
                    event = event_data['event']
                    action = event.get('action')
                    if action in ['closed', 'liquidated', 'loan_created']:
                        event_price = price_history[event_idx] if event_idx < len(price_history) else price_history[-1]
                        if action == 'liquidated':
                            marker_color, marker_style, marker_size = '#FF0000', 'X', 8
                            event_text = 'liquidated'
                            edge_width = 1.5
                        elif action == 'closed':
                            marker_color, marker_style, marker_size = '#FF0000', 'o', 7
                            event_text = 'loan closed'
                            edge_width = 1
                        else:  # loan_created
                            marker_color, marker_style, marker_size = '#1E90FF', 'D', 7
                            event_text = 'loan created'
                            edge_width = 1

                        ax_main.plot(
                            event_idx, event_price, marker_style,
                            color=marker_color, markersize=marker_size,
                            markeredgecolor=marker_color,
                            markeredgewidth=edge_width,
                            zorder=10, label="_nolegend_")
                        ax_main.text(
                            event_idx, event_price + (y_max - y_min) * 0.05,
                            event_text, fontsize=8, color=marker_color,
                            fontweight='bold', ha='center', va='bottom', zorder=6,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      edgecolor=marker_color, linewidth=1.0, alpha=0.9))
                    else:
                        style = get_event_style(action)
                        ax_main.axvline(
                            x=event_idx, color=style["color"],
                            linestyle=style["linestyle"],
                            linewidth=style["linewidth"],
                            alpha=0.6, zorder=2)

                    # Add to legend dynamically on first occurrence
                    if action not in legend_seen_actions and action not in {'closed', 'liquidated', 'loan_created'}:
                        legend_seen_actions.add(action)
                        label = EVENT_LABELS.get(action, action.replace('_', ' ').title())
                        style = get_event_style(action)
                        event_legend_handles.append(
                            Line2D([], [], color=style["color"],
                                   linestyle=style["linestyle"],
                                   linewidth=style["linewidth"], alpha=0.8))
                        event_legend_labels.append(label)
                        ax_main.legend(
                            base_legend_handles + event_legend_handles,
                            base_legend_labels + event_legend_labels,
                            loc='upper left')

            event_lines = []
            for eidx, elist in events_by_block_index.items():
                if eidx > idx:
                    continue
                for ed in elist:
                    event_lines.append(_fmt_event_line(ed['event'], ed))
            if len(event_lines) > 5:
                event_lines = event_lines[-5:]
            if event_lines:
                header = f"{'':1} {'Event':<14} {'Block':>10} {'Amount':>14} {'Health':>8}"
                separator = "-" * len(header)
                events_text_artist.set_text("\n".join([header, separator] + event_lines))
            else:
                events_text_artist.set_text('')
        else:
            events_text_artist.set_text('')

        health_fill_color, health_edge_color = get_health_color(health)
        health_bar_artist.set_height(health)
        health_bar_artist.set_color(health_fill_color)
        health_bar_artist.set_edgecolor(health_edge_color)

        if health >= 0:
            text_y = health + (health_y_max - health_y_min) * 0.02
            va = 'bottom'
        else:
            text_y = health - (health_y_max - health_y_min) * 0.02
            va = 'top'
        health_text_artist.set_text(f'{health:.2f}')
        health_text_artist.set_position((0, text_y))
        health_text_artist.set_va(va)
        health_text_artist.set_color(health_edge_color)

        current_debt = block_data.get('debt', 0) or 0
        current_collateral = block_data.get('total_collateral_value') or 0
        current_native = block_data.get('collateral_native', 0)
        current_crvusd = block_data.get('collateral_crvusd', 0)
        current_bands_state = block_data.get('num_bands_state', len(bands))

        ltv = (current_debt / current_collateral) if current_collateral > 0 else None
        ltv_text = f"{ltv*100:,.2f}%" if ltv is not None else "N/A"

        stats_text_artist.set_text("\n".join([
            f"Bands:       {current_bands_state}",
            f"LTV:         {ltv_text}",
            f"Debt:        {current_debt:,.2f}",
            f"Collateral:  ${current_collateral:,.2f}",
            f"  - {collateral_symbol:<6} {current_native:,.4f}",
            f"  - crvUSD:  {current_crvusd:,.4f}",
        ]))

        fig.savefig(frames_dir / f"{file_str}_{cycle}.png", dpi=dpi_num)

        if close_loan_block_index is not None and idx == close_loan_block_index:
            log_fn(f"  Stopping at close/liquidation event (block {block})")
            break

    plt.close(fig)

    # --- Encode video -------------------------------------------------------
    image_elapsed = time.time() - image_start_time
    log_fn(f"\n  Rendering: {image_elapsed:.1f}s")

    cmd = [
        "ffmpeg", "-y",
        "-start_number", "1",
        "-framerate", "50",
        "-i", str(frames_dir / f"{file_str}_%d.png"),
        "-vf", "scale=1280:-2:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        "-movflags", "+faststart",
        str(mp4_filename),
    ]
    result = subprocess.run(cmd)

    if result.returncode == 0 and mp4_filename.exists():
        shutil.rmtree(frames_dir)
        log_fn(f"  Cleaned up frames")
    else:
        log_fn(f"  ffmpeg failed (exit {result.returncode}), keeping frames in {frames_dir}")

    log_fn(f"  Output: {mp4_filename}")
    return mp4_filename


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python rerender.py <path/to/block_data.json>")
        sys.exit(1)
    rerender(sys.argv[1])
