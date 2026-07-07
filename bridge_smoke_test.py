"""
Bridge smoke test - no gymnasium/stable-baselines3 dependency.

Run this FIRST, before trusting terraria_env.py or terraria_hrl_env.py.
It just connects, prints one parsed GameState so you can eyeball that the
shapes/fields match what RLStateSystem.cs is actually sending, then lets you
optionally fire a single craft request interactively.

Usage:
    python bridge_smoke_test.py
    (then alt-tab into Terraria with the world loaded and the mod enabled)
"""

import socket
import json

HOST, PORT = "127.0.0.1", 7778


def recv_one_line(conn, buffer):
    while "\n" not in buffer:
        data = conn.recv(65536)
        if not data:
            raise ConnectionError("Game closed the connection")
        buffer += data.decode("utf-8")
    line, _, buffer = buffer.partition("\n")
    return json.loads(line), buffer


def describe(state: dict):
    print("\n--- GameState snapshot ---")
    print(f"Position:      ({state['PlayerX']:.1f}, {state['PlayerY']:.1f})")
    print(f"Health:        {state['Health']}/{state['MaxHealth']}")
    print(f"Velocity:      ({state['VelocityX']:.2f}, {state['VelocityY']:.2f})")

    grid = state.get("Grid")
    print(f"Grid:          {len(grid)} cells" if grid else "Grid:          MISSING")

    entities = state.get("Entities")
    if entities is not None:
        active = [e for e in entities if e["NpcType"] >= 0]
        print(f"Entities:      {len(entities)} slots, {len(active)} occupied")
        for e in active:
            print(f"   type={e['NpcType']} rel=({e['RelX']:.0f},{e['RelY']:.0f}) "
                  f"life={e['LifeFrac']:.2f} boss={e['IsBoss']}")
    else:
        print("Entities:      MISSING (are you still on the old RLStateSystem.cs?)")

    biome = state.get("Biome")
    print(f"Biome flags:   {biome}" if biome is not None else "Biome flags:   MISSING")

    inv = state.get("Inventory")
    print(f"Inventory:     {inv}" if inv is not None else "Inventory:     MISSING")

    milestones = state.get("Milestones")
    print(f"Milestones:    {milestones}" if milestones is not None else "Milestones:    MISSING")

    print(f"LastCraftResult: {state.get('LastCraftResult', 'MISSING')}")
    print("---------------------------\n")


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"Waiting for Terraria to connect on {HOST}:{PORT} ...")
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"Connected: {addr}\n")

    buffer = ""
    state, buffer = recv_one_line(conn, buffer)
    describe(state)

    item_id = input("Enter an ItemID to test-craft (blank to skip): ").strip()
    if item_id:
        cmd = {"Move": 0, "Jump": False, "UseItem": False, "Reset": False,
                "CraftItemID": int(item_id), "CraftRequestId": 1}
        conn.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        print("Sent craft request, watching LastCraftResult for the next few ticks...")
        for _ in range(10):
            state, buffer = recv_one_line(conn, buffer)
            result = state.get("LastCraftResult", 0)
            if result != 0:
                print(f"LastCraftResult = {result} "
                      f"(1=success, -1=no recipe, -2=missing ingredients, -3=inventory full)")
                break

    conn.close()
    srv.close()


if __name__ == "__main__":
    main()
