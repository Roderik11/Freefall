using System;
using System.Text.Json.Serialization;
using System.ComponentModel;

namespace Freefall.Assets
{
    public abstract class Asset
    {
        public string Name { get; set; }
        
        [Browsable(false)]
        public string AssetPath { get; set; }

        /// <summary>
        /// Sub-asset GUID assigned during import. Used by AssetFile
        /// serialization to write asset references as GUID strings.
        /// </summary>
        [JsonIgnore]
        [ReadOnly(true)]
        [Browsable(false)]
        public string Guid { get; set; }

        // ── Dirty Tracking ──

        /// <summary>
        /// True if this asset has been modified since last save.
        /// Set automatically by the inspector when properties change.
        /// </summary>
        [JsonIgnore]
        [Browsable(false)]
        public bool IsDirty { get; private set; }

        /// <summary>
        /// Mark this asset as modified. Called by the inspector on property changes.
        /// </summary>
        public virtual void MarkDirty()
        {
            IsDirty = true;
            Freefall.Base.MessageDispatcher.Send("AssetDirty", this);
        }

        /// <summary>
        /// Clear the dirty flag. Called after saving.
        /// </summary>
        public void ClearDirty() => IsDirty = false;

        // ── Streaming State ──
        private long _readyFenceValue = long.MaxValue;
        
        public bool IsReady(long completedFence)
        {
            return _readyFenceValue <= completedFence;
        }
        
        public void SetReadyFence(long fenceValue)
        {
            _readyFenceValue = fenceValue;
        }

        public void MarkReady()
        {
            _readyFenceValue = 0;
        }
        
        public Asset()
        {
             _readyFenceValue = 0;
        }
    }
}
