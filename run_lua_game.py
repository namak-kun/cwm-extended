"""Track A: trace a realistic, self-contained Lua GAME TICK with CWM.

Tests assumption A1 (symbolic state sufficient) + long-horizon game-state tracking
on REAL game structure (player, chase-AI enemies, collisions, scheduled spawns,
hp/score, game-over) in the actual game-scripting language. Deterministic (no RNG)
to isolate state-tracking from the aleatoric wall (A8).

Also documents A5: a real love2d `love.update` is engine-interleaved and does NOT
run under plain lua5.4 (engine functions). We trace the clean self-contained game.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

# A realistic arena game. simulate() runs the whole game; CWM traces the long
# multi-tick rollout and we compare the final checksum.
ARENA_LUA = r'''-- self-contained arena game (no love2d, runs under lua5.4)
local W, H = 6, 6

local function sign(a) if a > 0 then return 1 elseif a < 0 then return -1 else return 0 end end

local function step(state, action)
    -- move player by action
    local p = state.player
    if action == "U" then p.y = p.y - 1
    elseif action == "D" then p.y = p.y + 1
    elseif action == "L" then p.x = p.x - 1
    elseif action == "R" then p.x = p.x + 1 end
    if p.x < 1 then p.x = 1 elseif p.x > W then p.x = W end
    if p.y < 1 then p.y = 1 elseif p.y > H then p.y = H end
    -- player stomps any enemy it stepped onto
    for _, e in ipairs(state.enemies) do
        if e.alive and e.x == p.x and e.y == p.y then
            e.alive = false
            state.score = state.score + 10
        end
    end
    -- enemies chase the player one cell (greedy on larger axis gap)
    for _, e in ipairs(state.enemies) do
        if e.alive then
            local dx, dy = p.x - e.x, p.y - e.y
            if math.abs(dx) >= math.abs(dy) then e.x = e.x + sign(dx)
            else e.y = e.y + sign(dy) end
            if e.x == p.x and e.y == p.y then
                p.hp = p.hp - 1
            end
        end
    end
    state.tick = state.tick + 1
    -- scheduled spawn (deterministic): every 3 ticks add an enemy at a fixed slot
    if state.tick % 3 == 0 then
        local slot = state.tick // 3
        local sx = ((slot * 2) % W) + 1
        local sy = ((slot * 3) % H) + 1
        table.insert(state.enemies, {x = sx, y = sy, alive = true})
    end
    return state
end

function simulate()  -- << START_OF_TRACE
    local state = {
        player = {x = 3, y = 3, hp = 5, score = 0},
        enemies = {{x = 1, y = 1, alive = true}, {x = 6, y = 6, alive = true}},
        tick = 0,
    }
    local actions = {"R", "D", "R", "U", "L", "D", "R", "U"}
    for _, a in ipairs(actions) do
        state = step(state, a)
        if state.player.hp <= 0 then break end
    end
    local alive = 0
    for _, e in ipairs(state.enemies) do if e.alive then alive = alive + 1 end end
    return state.player.score * 1000 + state.player.hp * 100 + alive * 10
           + state.player.x + state.player.y
end

print(simulate())
'''

# A REAL love2d update fn (illustrative) — calls engine funcs, NOT runnable under lua5.4.
LOVE2D_SNIPPET = r'''function love.update(dt)
    player.x = player.x + player.vx * dt
    if love.keyboard.isDown("right") then player.vx = 200 end
    for i, b in ipairs(bullets) do
        b.y = b.y - b.speed * dt
        if b.y < 0 then table.remove(bullets, i) end
    end
    enemySpawnTimer = enemySpawnTimer - dt
    if enemySpawnTimer <= 0 then
        spawnEnemy(love.math.random(0, love.graphics.getWidth()))
        enemySpawnTimer = 1.5
    end
end
'''


def lua_ground_truth(src: str):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "g.lua"); open(f, "w").write(src)
        r = subprocess.run(["lua5.4", f], capture_output=True, text=True, timeout=60)
        out = r.stdout.strip()
        try:
            return int(out.splitlines()[-1])
        except (ValueError, IndexError):
            return out or r.stderr.strip()[:120]


def check_love2d_runs():
    """Demonstrate A5: the real love2d snippet doesn't run under lua5.4."""
    wrapped = LOVE2D_SNIPPET + "\nlove.update(0.1)\n"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "l.lua"); open(f, "w").write(wrapped)
        r = subprocess.run(["lua5.4", f], capture_output=True, text=True)
        return r.returncode, (r.stderr.strip()[:140] or "ran")


def main_run(model_path, tp=2):
    from models.cwm_trace import (CWMvLLM, Event, CALL_SEP, build_prompt, parse_full_trace, resolve_locals)
    from run_ood import cwm_final_return

    t0 = time.time()
    gt = lua_ground_truth(ARENA_LUA)
    rc, msg = check_love2d_runs()
    print(f"A5 check: real love2d snippet under lua5.4 -> returncode={rc}, msg={msg!r}", flush=True)
    print(f"arena game ground-truth checksum = {gt}", flush=True)

    m = CWMvLLM(model_path, tp=tp, max_model_len=16384)
    print("== CWM loaded ==", flush=True)

    prompt = build_prompt(m, ARENA_LUA, [], force_event=Event.CALL)
    gen = m.gen_full_trace_tokens(prompt, max_tokens=14000)
    pred = parse_full_trace(m, [CALL_SEP] + gen)
    cv = cwm_final_return(pred)
    print(f"\n=== Lua ARENA GAME: true={gt}  cwm={cv}  MATCH={cv == gt}  frames={len(pred)} ===")
    # dump a window around each game tick boundary to see state tracking
    for i, f in enumerate(pred):
        if f.source_line.strip().startswith("state = step") or "return state.player.score" in f.source_line or i < 6:
            print(f"  [{i:3}] {f.event.name:7} {f.source_line.strip()[:46]:46} "
                  f"player={resolve_locals(f).get('state','')[:60] if isinstance(resolve_locals(f).get('state'), str) else ''} arg={f.arg}")

    out = {"arena_true": gt, "arena_cwm": cv, "match": cv == gt, "frames": len(pred),
           "love2d_a5_returncode": rc, "love2d_a5_msg": msg, "elapsed_sec": round(time.time()-t0, 1)}
    json.dump(out, open("results/cwm_lua_game.json", "w"), indent=2)
    print(f"\nsaved -> results/cwm_lua_game.json ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main_run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
