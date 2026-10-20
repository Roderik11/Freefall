using System;
using System.Collections.Generic;
using System.Numerics;

namespace Freefall
{
    /// <summary>
    /// Input a tool plays into the game instead of the player's hands (the editor's MCP play_input):
    /// keys and mouse buttons down and up, waits, mouse look, the game UI's cursor set or moved, the
    /// wheel. Steps that take no time run one after another within a frame; a wait, a move or a look
    /// ends the frame's share and goes on in the next ones. <see cref="Input.Play"/> starts one;
    /// <see cref="Input.Update"/> plays it before anything reads the input, so the game sees it
    /// exactly as it sees the devices.
    /// </summary>
    public sealed class InputScript
    {
        public enum Kind
        {
            /// <summary>A key goes down (Key).</summary>
            KeyDown,
            /// <summary>A key comes up (Key).</summary>
            KeyUp,
            /// <summary>A mouse button goes down (Button: 0 left, 1 right, 2 middle).</summary>
            ButtonDown,
            /// <summary>A mouse button comes up.</summary>
            ButtonUp,
            /// <summary>Nothing for Seconds; 0 = until the next frame.</summary>
            Wait,
            /// <summary>The game UI's cursor goes to To (pixels of the game's picture from its top left).</summary>
            Cursor,
            /// <summary>The game UI's cursor is the real mouse's again.</summary>
            ReleaseCursor,
            /// <summary>The game UI's cursor glides from where it is to To over Seconds.</summary>
            Move,
            /// <summary>The mouse is moved by To (raw counts: the camera turns) spread over Seconds.</summary>
            Look,
            /// <summary>The wheel turns by Amount (120 = one notch).</summary>
            Wheel,
        }

        public struct Step
        {
            public Kind Kind;
            public Keys Key;
            public int Button;
            public Vector2 To;
            public float Seconds;
            public int Amount;

            public override string ToString() => Kind switch
            {
                Kind.KeyDown or Kind.KeyUp => $"{Kind} {Key}",
                Kind.ButtonDown or Kind.ButtonUp => $"{Kind} {Button}",
                Kind.Wait => $"Wait {Seconds:0.##}s",
                Kind.Cursor => $"Cursor {To.X:0},{To.Y:0}",
                Kind.Move => $"Move {To.X:0},{To.Y:0} over {Seconds:0.##}s",
                Kind.Look => $"Look {To.X:0},{To.Y:0} over {Seconds:0.##}s",
                Kind.Wheel => $"Wheel {Amount}",
                _ => Kind.ToString(),
            };
        }

        private readonly List<Step> _steps;
        private int _index;
        private bool _started;
        private float _elapsed;
        private Vector2 _from, _looked;

        public InputScript(IEnumerable<Step> steps) => _steps = new List<Step>(steps);

        public int Count => _steps.Count;

        /// <summary>The step playing now (0-based); Count once it is done.</summary>
        public int Index => _index;

        public bool Done => _index >= _steps.Count;

        /// <summary>The step playing now, for whoever watches.</summary>
        public string Current => Done ? "done" : _steps[_index].ToString();

        /// <summary>Seconds the steps take, waits, moves and looks together (a frame for each 0-second wait not counted).</summary>
        public float Seconds
        {
            get
            {
                float total = 0f;
                foreach (var step in _steps)
                    if (step.Kind is Kind.Wait or Kind.Move or Kind.Look) total += step.Seconds;
                return total;
            }
        }

        /// <summary>One frame of it. Called by Input.Update.</summary>
        internal void Tick(float delta)
        {
            while (!Done)
            {
                var step = _steps[_index];
                switch (step.Kind)
                {
                    case Kind.KeyDown: Input.SimulateKey(step.Key, true); break;
                    case Kind.KeyUp: Input.SimulateKey(step.Key, false); break;
                    case Kind.ButtonDown: Input.SimulateButton(step.Button, true); break;
                    case Kind.ButtonUp: Input.SimulateButton(step.Button, false); break;
                    case Kind.Cursor: Input.SimulateCursor(step.To); break;
                    case Kind.ReleaseCursor: Input.SimulateCursor(null); break;
                    case Kind.Wheel: Input.SimulateWheel(step.Amount); break;

                    case Kind.Wait:
                        // The frame it begins in is its first; it ends once its time has gone by
                        if (!_started)
                        {
                            _started = true;
                            _elapsed = 0f;
                            return;
                        }
                        _elapsed += delta;
                        if (_elapsed < step.Seconds) return;
                        break;

                    case Kind.Move:
                    {
                        if (!_started)
                        {
                            _started = true;
                            _elapsed = 0f;
                            var at = Input.SimulatedCursor;
                            _from = at.HasValue ? new Vector2(at.Value.X, at.Value.Y) : step.To;
                        }
                        else
                            _elapsed += delta;
                        float t = step.Seconds > 0f ? Math.Clamp(_elapsed / step.Seconds, 0f, 1f) : 1f;
                        Input.SimulateCursor(Vector2.Lerp(_from, step.To, t));
                        if (t < 1f) return;
                        break;
                    }

                    case Kind.Look:
                    {
                        if (!_started)
                        {
                            _started = true;
                            _elapsed = 0f;
                            _looked = Vector2.Zero;
                        }
                        _elapsed += delta;
                        float t = step.Seconds > 0f ? Math.Clamp(_elapsed / step.Seconds, 0f, 1f) : 1f;
                        var goal = step.To * t;
                        var move = new Vector2(MathF.Round(goal.X - _looked.X), MathF.Round(goal.Y - _looked.Y));
                        Input.SimulateLook((int)move.X, (int)move.Y);
                        _looked += move;
                        if (t < 1f) return;
                        break;
                    }
                }

                _index++;
                _started = false;
            }
        }
    }
}
