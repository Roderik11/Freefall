using System.Collections.Generic;
using Freefall.Assets;

namespace Freefall.Graphics
{
    // ── A material's on-demand textures (Texture.OnDemand) ──
    // A material holds its on-demand textures while something draws with it: every persistent draw registered with it
    // (CommandBuffer.AddPersistent: MeshRenderer, SkinnedMeshRenderer, cloth, everything a character wears or carries),
    // per-frame draws and direct applies by a lease renewed on every use (Touch), and the few systems that write its
    // MaterialID into their own buffers by name (decals, terrain decorations, tube trails, icons). Its slots bind the
    // texture once the copy has arrived and a fallback before that and whenever nobody holds it: so a material never
    // keeps the bindless index of a copy that may go. rendering_engine/performance/vram.md "Textures only while drawn".
    public partial class Material
    {
        private int _holders;
        private readonly List<Texture> _heldTextures = new();   // the on-demand textures this material acquired, one per hold
        private int _leaseTick = int.MinValue;
        private bool _leased;
        private bool _hasDemandTextures;   // a slot holds an on-demand texture: only then does a lease do anything

        /// <summary>How many hold this material (persistent draws, a lease, decals, icons).</summary>
        public int Holders => _holders;

        /// <summary>
        /// Something draws with this material from now on: its on-demand textures are loaded (on workers) and bound as they
        /// arrive, a fallback meanwhile. Pair every call with <see cref="Release"/>. Any thread.
        /// </summary>
        public void Acquire()
        {
            lock (OnDemandTextures.Sync)
                if (_holders++ == 0) HoldTextures();
        }

        /// <summary>
        /// One holder fewer. With none left the slots bind their fallbacks and the textures go idle (dropped beyond
        /// <see cref="OnDemandTextures.IdleBudget"/>). Any thread.
        /// </summary>
        public void Release()
        {
            lock (OnDemandTextures.Sync)
            {
                if (_holders <= 0)
                {
                    Debug.LogWarning("Material", $"'{Name}' released more often than acquired: ignored");
                    return;
                }
                if (--_holders == 0) LetGoOfTextures();
            }
        }

        /// <summary>
        /// Drawn this frame by a path that registers nothing (a per-frame CommandBuffer.Enqueue, a direct Apply, a tube
        /// trail): held until it goes <see cref="OnDemandTextures.LeaseTicks"/> frames without one. Free when the material
        /// has no on-demand texture or was touched this frame already. Any thread.
        /// </summary>
        public void Touch()
        {
            if (!_hasDemandTextures) return;
            int tick = Engine.TickCount;
            if (_leaseTick == tick) return;
            _leaseTick = tick;
            if (_leased) return;
            lock (OnDemandTextures.Sync)
            {
                if (_leased) return;
                _leased = true;
                if (_holders++ == 0) HoldTextures();
                OnDemandTextures.Lease(this);
            }
        }

        internal bool LeaseExpired(int tick) => tick - _leaseTick > OnDemandTextures.LeaseTicks;

        // The lease ran out (OnDemandTextures.Update, under the lock)
        internal void EndLease()
        {
            _leased = false;
            if (--_holders == 0) LetGoOfTextures();
        }

        // Under the lock
        private void HoldTextures()
        {
            foreach (var texture in _textures.Values)
            {
                if (!texture.OnDemand) continue;
                texture.AcquireFor(this);
                _heldTextures.Add(texture);
            }
            RefreshTextureIndices();
        }

        private void LetGoOfTextures()
        {
            foreach (var texture in _heldTextures)
                texture.ReleaseFor(this);
            _heldTextures.Clear();
            RefreshTextureIndices();
        }

        // The slots changed (under the lock): while held, hold the new set, then let go of the old one, so a texture in both
        // is never let go of in between
        private void SyncHeldTextures()
        {
            _hasDemandTextures = false;
            foreach (var texture in _textures.Values)
                _hasDemandTextures |= texture.OnDemand;
            if (_holders == 0) return;

            var old = _heldTextures.ToArray();
            _heldTextures.Clear();
            foreach (var texture in _textures.Values)
            {
                if (!texture.OnDemand) continue;
                texture.AcquireFor(this);
                _heldTextures.Add(texture);
            }
            foreach (var texture in old)
                texture.ReleaseFor(this);
        }

        /// <summary>
        /// Write every texture slot as it stands: the texture where it is resident (always, or arrived while held), its
        /// fallback otherwise. Called when a held copy arrives or goes, and when holds begin or end. Under the lock.
        /// </summary>
        internal void RefreshTextureIndices()
        {
            foreach (var (name, texture) in _textures)
            {
                UpdateMaterialData(name, SlotIndex(name, texture));
                if (_effectTextureSettings.TryGetValue(name, out var setting))
                    WriteSetting(setting.Offset, SettingIndex(setting, texture));
            }
        }

        /// <summary>Whether a texture can be bound now: resident for good, or held by this material and arrived.</summary>
        private bool Bindable(Texture texture) => !texture.OnDemand || (_holders > 0 && texture.Arrived);

        /// <summary>
        /// What a MaterialData slot (or an effect's push-constant texture) binds for this texture. Until its copy is there:
        /// the albedo the pending texture (dark grey with no alpha: opaque surfaces show grey, alpha-tested ones such as
        /// people and leaves nothing), every other slot 0, which the shaders read as "no map" (roughness 0.65, no metal,
        /// no AO, no emission, the mesh's normal).
        /// </summary>
        private uint SlotIndex(string name, Texture texture)
        {
            if (Bindable(texture)) return texture.BindlessIndex;
            return name is "Albedo" or "AlbedoTex" ? PendingIndex : 0u;
        }

        /// <summary>
        /// What an effect's texture setting (TextureRef) binds: the texture once it can be, the effect's declared default
        /// when none is set; until a set texture's copy is there, a flat normal where the default is one, else the pending
        /// texture (no alpha, so what it masks or erodes is not there yet). Never 0: the shaders read these unchecked.
        /// </summary>
        private uint SettingIndex(MaterialParamDesc setting, Texture? texture)
        {
            var fallback = setting.DefaultTextureAsset;
            if (texture == null) return fallback?.BindlessIndex ?? 0u;
            if (Bindable(texture)) return texture.BindlessIndex;
            if (fallback != null && fallback == InternalAssets.FlatNormal) return fallback.BindlessIndex;
            return PendingIndex;
        }

        private static uint PendingIndex => InternalAssets.Pending?.BindlessIndex ?? InternalAssets.Gray?.BindlessIndex ?? 0u;
    }
}
