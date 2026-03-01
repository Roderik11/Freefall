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
        
        // Streaming State
        private long _readyFenceValue = long.MaxValue; // Default to 'never ready' until set? Or 0 if loaded synchronously?
        
        // If loaded synchronously (old path), this should be 0 (always ready).
        // If async, it starts at MaxValue, then gets set to a real fence value.
        
        public bool IsReady(long completedFence)
        {
            return _readyFenceValue <= completedFence;
        }
        
        public void SetReadyFence(long fenceValue)
        {
            _readyFenceValue = fenceValue;
        }

        // Helper for synchronous loading legacy support
        public void MarkReady()
        {
            _readyFenceValue = 0;
        }
        
        public Asset()
        {
             // By default, assume ready (legacy compat). 
             // Async loaders must explicitly set SetReadyFence(long.MaxValue) or similar at start.
             _readyFenceValue = 0;
        }
    }
}
