using Terraria;
using Terraria.ModLoader;
using Terraria.ID;
using Microsoft.Xna.Framework;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;

namespace TerrariaRLBridge
{
    // -----------------------------------------------------------------------
    // PART 1: THE BRAIN (Networking)
    // -----------------------------------------------------------------------
    public class RLStateSystem : ModSystem
    {
        // Shared Data (Static so the Player class can read it)
        public static bool IsConnected = false;
        public static AgentAction CurrentAction = new AgentAction();
        public static bool HasLoggedOverride = false;

        // --- Craft "cheat" API shared state ---
        // CraftRequestId is used to edge-trigger crafting: without it, holding the
        // same CraftItemID across multiple ticks (which WILL happen, since
        // CurrentAction persists until a new packet arrives) would re-craft the
        // item every single tick instead of once per request.
        public static int LastHandledCraftRequestId = 0;
        public static int LastCraftResult = 0; // 0=idle, 1=success, -1=no recipe, -2=missing ingredients, -3=inventory full

        private const string IP_ADDRESS = "127.0.0.1";
        private const int PORT = 7778;

        private static TcpClient _client;
        private static NetworkStream _stream;
        private string _packetBuffer = "";
        private int _retryCounter = 0;

        // How many of the closest entities to report each tick.
        private const int MAX_ENTITIES = 8;
        private const float ENTITY_RANGE = 800f; // pixels (50 tiles)

        public struct GameState
        {
            public float PlayerX;
            public float PlayerY;
            public float VelocityX;
            public float VelocityY;
            public int Health;
            public int MaxHealth;
            public int[] Grid;

            // --- Phase 5.1 additions ---
            public float[] Biome;        // fixed-order zone flags, see GetBiomeFlags()
            public EntityInfo[] Entities; // closest MAX_ENTITIES active, non-town NPCs
            public float[] Inventory;    // bucketed inventory features, see GetInventoryFeatures()
            public bool[] Milestones;    // boss-downed / progression flags, see GetMilestones()
            public int LastCraftResult;  // result of the most recently handled craft request
        }

        public struct EntityInfo
        {
            public int NpcType;   // -1 = empty slot (padding, fewer than MAX_ENTITIES nearby)
            public float RelX;    // position relative to the player, in pixels
            public float RelY;
            public float LifeFrac; // life / lifeMax, already normalized to [0,1]
            public bool IsBoss;
            public int Damage;
        }

        public class AgentAction
        {
            public int Move;
            public bool Jump;
            public bool UseItem;
            public bool Reset;

            // --- Craft "cheat" API ---
            // Set CraftItemID to the desired ItemID and increment CraftRequestId
            // (any change in value works - a simple counter is easiest) to trigger
            // one craft attempt. Leaving CraftRequestId unchanged will not re-fire.
            public int CraftItemID;
            public int CraftRequestId;
        }

        // --- CONNECTION LIFECYCLE ---
        public override void OnWorldLoad()
        {
            _retryCounter = 0;
            ConnectToBrain();
        }

        public override void OnWorldUnload()
        {
            CloseConnection();
        }

        // --- UPDATE LOOP ---
        public override void PostUpdateEverything()
        {
            if (Main.LocalPlayer == null || !Main.LocalPlayer.active) return;

            if (IsConnected)
            {
                SendState(ExtractState());
                ReceiveAction();
            }
            else
            {
                _retryCounter++;
                if (_retryCounter >= 60)
                {
                    _retryCounter = 0;
                    ConnectToBrain();
                }
            }
        }

        // --- NETWORKING INTERNALS ---
        private void ConnectToBrain()
        {
            try
            {
                _client = new TcpClient();
                _client.BeginConnect(IP_ADDRESS, PORT, new AsyncCallback(ConnectCallback), _client);
            }
            catch (Exception) { }
        }

        private void ConnectCallback(IAsyncResult ar)
        {
            try
            {
                if (_client == null) return;
                TcpClient t = (TcpClient)ar.AsyncState;
                t.EndConnect(ar);
                t.NoDelay = true; // disable Nagle - we send small packets every tick and want low latency, not throughput
                _stream = t.GetStream();
                IsConnected = true;
                HasLoggedOverride = false;
                Mod.Logger.Info("[RL-Bridge] Connected to Brain on Port 7778!");
            }
            catch (Exception) { IsConnected = false; }
        }

        private void SendState(GameState state)
        {
            try
            {
                string json = JsonConvert.SerializeObject(state);
                byte[] data = Encoding.UTF8.GetBytes(json + "\n");
                _stream.Write(data, 0, data.Length);
            }
            catch (Exception e)
            {
                Mod.Logger.Warn($"[RL-Bridge] Send Failed: {e.Message}");
                CloseConnection();
            }
        }

        private void ReceiveAction()
        {
            try
            {
                if (_stream.DataAvailable)
                {
                    byte[] readBuffer = new byte[4096];
                    int bytesRead = _stream.Read(readBuffer, 0, readBuffer.Length);

                    if (bytesRead > 0)
                    {
                        string chunk = Encoding.UTF8.GetString(readBuffer, 0, bytesRead);
                        _packetBuffer += chunk;

                        while (_packetBuffer.Contains("\n"))
                        {
                            int splitIndex = _packetBuffer.IndexOf('\n');
                            string message = _packetBuffer.Substring(0, splitIndex);
                            _packetBuffer = _packetBuffer.Substring(splitIndex + 1);

                            if (!string.IsNullOrWhiteSpace(message))
                            {
                                try
                                {
                                    CurrentAction = JsonConvert.DeserializeObject<AgentAction>(message);
                                }
                                catch { }
                            }
                        }
                    }
                }
            }
            catch (Exception) { CloseConnection(); }
        }

        private void CloseConnection()
        {
            if (!IsConnected) return;
            IsConnected = false;
            if (_stream != null) _stream.Close();
            if (_client != null) _client.Close();
            _stream = null;
            _client = null;
            _packetBuffer = "";
            Mod.Logger.Info("[RL-Bridge] Connection Closed.");
        }

        // --- STATE EXTRACTION ---
        private GameState ExtractState()
        {
            Player p = Main.LocalPlayer;
            int radius = 20;
            return new GameState
            {
                PlayerX = p.Center.X,
                PlayerY = p.Center.Y,
                VelocityX = p.velocity.X,
                VelocityY = p.velocity.Y,
                Health = p.statLife,
                MaxHealth = p.statLifeMax2,
                Grid = GetLocalGrid(p, radius),
                Biome = GetBiomeFlags(p),
                Entities = GetNearbyEntities(p),
                Inventory = GetInventoryFeatures(p),
                Milestones = GetMilestones(),
                LastCraftResult = LastCraftResult
            };
        }

        private int[] GetLocalGrid(Player p, int radius)
        {
            int centerTileX = (int)(p.Center.X / 16f);
            int centerTileY = (int)(p.Center.Y / 16f);
            int size = radius * 2 + 1;
            int[] grid = new int[size * size];
            int counter = 0;

            for (int y = centerTileY - radius; y <= centerTileY + radius; y++)
            {
                for (int x = centerTileX - radius; x <= centerTileX + radius; x++)
                {
                    if (WorldGen.InWorld(x, y))
                    {
                        Tile tile = Main.tile[x, y];
                        int value = 0;
                        if (tile.HasTile)
                        {
                            if (TileID.Sets.TouchDamageImmediate[tile.TileType] > 0) value = 3;
                            else if (Main.tileSolidTop[tile.TileType]) value = 2;
                            else if (Main.tileSolid[tile.TileType]) value = 1;
                        }
                        grid[counter] = value;
                    }
                    else { grid[counter] = 1; }
                    counter++;
                }
            }
            return grid;
        }

        // Fixed-order biome/zone booleans, cast to 0f/1f so the whole thing is a
        // plain float vector on the Python side. Order matters - keep it stable
        // once you start training, since the Python env will index into this
        // array positionally.
        // NOTE: ZoneHallow was named ZoneHoly in older tModLoader/Terraria versions -
        // if this doesn't compile, that's almost certainly why. Check your version.
        private float[] GetBiomeFlags(Player p)
        {
            return new float[]
            {
                p.ZoneCorrupt ? 1f : 0f,
                p.ZoneCrimson ? 1f : 0f,
                p.ZoneHallow ? 1f : 0f,
                p.ZoneJungle ? 1f : 0f,
                p.ZoneSnow ? 1f : 0f,
                p.ZoneDesert ? 1f : 0f,
                p.ZoneDungeon ? 1f : 0f,
                p.ZoneUndergroundDesert ? 1f : 0f,
                p.ZoneGlowshroom ? 1f : 0f,
                p.ZoneMeteor ? 1f : 0f,
                p.ZoneBeach ? 1f : 0f,
                p.ZoneUnderworldHeight ? 1f : 0f,
                p.ZoneRockLayerHeight ? 1f : 0f,   // caverns
                p.ZoneDirtLayerHeight ? 1f : 0f,   // underground, above cavern layer
                p.ZoneOverworldHeight ? 1f : 0f,   // surface
                p.ZoneSkyHeight ? 1f : 0f,
            };
        }

        // Closest MAX_ENTITIES active, non-town NPCs within ENTITY_RANGE, sorted by
        // distance. Slots beyond however many were actually found are padded with
        // NpcType = -1 so the array is always a fixed size for the neural net.
        private EntityInfo[] GetNearbyEntities(Player p)
        {
            var candidates = new List<(float distSq, EntityInfo info)>();

            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC npc = Main.npc[i];
                if (npc == null || !npc.active || npc.life <= 0) continue;
                if (npc.townNPC) continue; // not combat/threat relevant for the Combat Gym

                float dx = npc.Center.X - p.Center.X;
                float dy = npc.Center.Y - p.Center.Y;
                float distSq = dx * dx + dy * dy;
                if (distSq > ENTITY_RANGE * ENTITY_RANGE) continue;

                candidates.Add((distSq, new EntityInfo
                {
                    NpcType = npc.type,
                    RelX = dx,
                    RelY = dy,
                    LifeFrac = npc.lifeMax > 0 ? (float)npc.life / npc.lifeMax : 0f,
                    IsBoss = npc.boss,
                    Damage = npc.damage
                }));
            }

            candidates.Sort((a, b) => a.distSq.CompareTo(b.distSq));

            var result = new EntityInfo[MAX_ENTITIES];
            for (int i = 0; i < MAX_ENTITIES; i++)
            {
                result[i] = i < candidates.Count ? candidates[i].info : new EntityInfo { NpcType = -1 };
            }
            return result;
        }

        // A small, hand-picked set of bucketed inventory stats rather than raw
        // item IDs (per the project's own "Item Grouping" rule). Treat the exact
        // item IDs here as a starting point - extend as your gyms need more signal
        // (e.g. HasGrapplingHook, HasWings, BestArmorDefense).
        private float[] GetInventoryFeatures(Player p)
        {
            int bestMeleeDamage = 0, bestRangedDamage = 0, bestMagicDamage = 0;
            bool hasPickaxe = false, hasAxe = false;
            int healthPotionCount = 0;
            long goldValue = 0;

            foreach (Item item in p.inventory)
            {
                if (item == null || item.type <= 0 || item.stack <= 0) continue;

                if (item.damage > 0)
                {
                    if (item.CountsAsClass(DamageClass.Melee)) bestMeleeDamage = Math.Max(bestMeleeDamage, item.damage);
                    else if (item.CountsAsClass(DamageClass.Ranged)) bestRangedDamage = Math.Max(bestRangedDamage, item.damage);
                    else if (item.CountsAsClass(DamageClass.Magic)) bestMagicDamage = Math.Max(bestMagicDamage, item.damage);
                }
                if (item.pick > 0) hasPickaxe = true;
                if (item.axe > 0) hasAxe = true;
                if (item.type == ItemID.LesserHealingPotion || item.type == ItemID.HealingPotion)
                    healthPotionCount += item.stack;

                goldValue += (long)item.value * item.stack;
            }

            return new float[]
            {
                bestMeleeDamage,
                bestRangedDamage,
                bestMagicDamage,
                hasPickaxe ? 1f : 0f,
                hasAxe ? 1f : 0f,
                healthPotionCount,
                goldValue / 100f // rough copper->silver-ish normalization, tune as needed
            };
        }

        // Boss-downed flags double as a built-in progression curriculum signal:
        // reward the Manager/Switchboard whenever one of these flips from false
        // to true, rather than hand-scripting "fight boss X after boss Y".
        private bool[] GetMilestones()
        {
            return new bool[]
            {
                NPC.downedBoss1,       // Eye of Cthulhu
                NPC.downedBoss2,       // Brain of Cthulhu / Eater of Worlds
                NPC.downedBoss3,       // Skeletron
                NPC.downedQueenBee,
                NPC.downedDeerclops,
                NPC.downedMechBossAny, // any of the three mech bosses
                NPC.downedPlantBoss,   // Plantera
                NPC.downedGolemBoss,
                NPC.downedFishron,
                NPC.downedEmpressOfLight,
                NPC.downedMoonlord,
            };
        }

        // --- CRAFT "CHEAT" API ---
        // Finds the first registered recipe that produces resultItemType, checks
        // the player's inventory for the required ingredients (tile/condition
        // requirements are deliberately NOT checked - that's the point of the
        // cheat), consumes them, and grants the result directly.
        //
        // KNOWN LIMITATION: this does not handle Recipe Groups (e.g. "any Iron Bar
        // or Lead Bar"), and if an item has multiple recipes it always tries the
        // first one tModLoader registered. Both are solvable later with
        // RecipeGroup.recipeGroups / recipe.acceptedGroupIndices if a given gym
        // needs it - flagged here rather than silently wrong.
        public static int TryInstantCraft(Player player, int resultItemType)
        {
            Recipe match = null;
            for (int r = 0; r < Recipe.numRecipes; r++)
            {
                Recipe recipe = Main.recipe[r];
                if (recipe?.createItem != null && recipe.createItem.type == resultItemType)
                {
                    match = recipe;
                    break;
                }
            }

            if (match == null) return -1; // no recipe produces this item

            var needed = new Dictionary<int, int>();
            foreach (Item ingredient in match.requiredItem)
            {
                if (ingredient.type == 0) break; // requiredItem is a fixed-size array; type 0 marks the end
                needed[ingredient.type] = needed.GetValueOrDefault(ingredient.type, 0) + ingredient.stack;
            }

            foreach (var kvp in needed)
            {
                int have = 0;
                foreach (Item slot in player.inventory)
                {
                    if (slot.type == kvp.Key) have += slot.stack;
                }
                if (have < kvp.Value) return -2; // insufficient ingredients
            }

            foreach (var kvp in needed)
            {
                int remaining = kvp.Value;
                foreach (Item slot in player.inventory)
                {
                    if (remaining <= 0) break;
                    if (slot.type != kvp.Key) continue;
                    int take = Math.Min(remaining, slot.stack);
                    slot.stack -= take;
                    if (slot.stack <= 0) slot.TurnToAir();
                    remaining -= take;
                }
            }

            player.QuickSpawnItem(player.GetSource_FromThis(), match.createItem.type, match.createItem.stack);
            return 1; // success. NOTE: we don't currently detect "inventory was full so it dropped on the ground" -
                      // if that distinction matters for your reward function, compare an inventory item-count
                      // checksum before/after instead of trusting this return value alone.
        }
    }

    // -----------------------------------------------------------------------
    // PART 2: THE BODY (Input Hijacking)
    // -----------------------------------------------------------------------
    public class RLInputPlayer : ModPlayer
    {
        public override void SetControls()
        {
            if (!RLStateSystem.IsConnected) return;

            if (!RLStateSystem.HasLoggedOverride)
            {
                Mod.Logger.Info("[RL-Player] OVERRIDING INPUTS VIA SETCONTROLS");
                RLStateSystem.HasLoggedOverride = true;
            }

            Player.controlLeft = false;
            Player.controlRight = false;
            Player.controlJump = false;
            Player.controlUseItem = false;
            Player.controlDown = false;
            Player.controlUp = false;
            Player.controlHook = false;
            Player.controlMount = false;

            var action = RLStateSystem.CurrentAction;

            if (action.Reset)
            {
                Player.Teleport(new Vector2(Main.spawnTileX * 16, (Main.spawnTileY * 16) - 48), 1);
                Player.velocity = Vector2.Zero;
                return;
            }

            // Edge-triggered craft: only fires once per distinct CraftRequestId,
            // otherwise holding the same action across ticks would re-craft the
            // item every tick until a new packet arrives.
            if (action.CraftItemID > 0 && action.CraftRequestId != RLStateSystem.LastHandledCraftRequestId)
            {
                RLStateSystem.LastHandledCraftRequestId = action.CraftRequestId;
                RLStateSystem.LastCraftResult = RLStateSystem.TryInstantCraft(Player, action.CraftItemID);
            }

            if (action.Move == -1) Player.controlLeft = true;
            if (action.Move == 1) Player.controlRight = true;
            if (action.Jump) Player.controlJump = true;
            if (action.UseItem) Player.controlUseItem = true;
        }
    }
}
