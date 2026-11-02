using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using PhysX;

namespace Freefall.Base
{
    public class PhysXErrorOutput : ErrorCallback
    {
        // A call PhysX refuses changes nothing and is reported only here, so warnings and errors
        // have to reach the console in Release builds too; Debug.Log is compiled out of those.
        public override void ReportError(ErrorCode errorCode, string message, string file, int lineNumber)
        {
            string text = $"[PhysX] {errorCode}: {message}";

            switch (errorCode)
            {
                case ErrorCode.NoError:
                case ErrorCode.DebugInfo:
                    Debug.Log(text);
                    break;
                case ErrorCode.Warning:
                    Debug.LogWarning(text);
                    break;
                default:
                    Debug.LogError(text);
                    break;
            }
        }
    }

    public class PhysXFilterShader : SimulationFilterShader
    {
        /// <summary>
        /// Word 0 of a shape's simulation filter data: a part of a ragdoll (Ragdoll), whose word 1 says which ragdoll.
        /// It meets nothing of its own ragdoll (the joints hold the parts together) and no kinematic actor (characters'
        /// capsules, door leaves): the living walk through a corpse rather than kicking it about.
        /// </summary>
        public const uint RagdollPart = 1u;

        // PxFilterObjectFlag::eKINEMATIC in the attributes of either side
        private const int Kinematic = 1 << 4;

        public override FilterResult Filter(int attributes0, FilterData filterData0, int attributes1, FilterData filterData1)
        {
            bool ragdoll0 = (filterData0.Word0 & RagdollPart) != 0, ragdoll1 = (filterData1.Word0 & RagdollPart) != 0;
            if ((ragdoll0 && ragdoll1 && filterData0.Word1 == filterData1.Word1)
                || (ragdoll0 && (attributes1 & Kinematic) != 0) || (ragdoll1 && (attributes0 & Kinematic) != 0))
                return new FilterResult { FilterFlag = FilterFlag.Kill, PairFlags = 0 };

            return new FilterResult
            {
                FilterFlag = FilterFlag.Default,
                PairFlags = PairFlag.ContactDefault
            };
        }
    }

    public static class PhysicsWorld
    {
        private const float FixedTimestep = 1f / 60f;
        private static float _accumulator;

        public static PhysX.Physics Physics { get; private set; }
        public static Scene Scene { get; private set; }
        public static ControllerManager ControllerManager { get; private set; }
        public static PhysX.Material DefaultMaterial { get; private set; }

        public static void Initialize()
        {
            // Before the first object: it learns every object's pointer as the wrapper registers it
            PhysXRelease.Initialize();

            var errorOutput = new PhysXErrorOutput();
            var foundation = new Foundation(errorOutput);

            // PVD — connect only when the PVD app is listening: a TCP connect to a closed localhost port retries for
            // ~2 s on Windows before it fails, which every editor start paid (2026-10-17)
            var pvd = new PhysX.VisualDebugger.Pvd(foundation);
            Physics = new PhysX.Physics(foundation, true, pvd);

            if (IsListening(PvdPort))
            {
                try
                {
                    Physics.Pvd.Connect("localhost");
                    Debug.Log("[PhysX] PVD connected");
                }
                catch
                {
                    Debug.Log("[PhysX] PVD not available (optional)");
                }
            }

            // Create scene with gravity and collision filtering
            var sceneDesc = new SceneDesc
            {
                Gravity = new Vector3(0, -9.81f, 0),
                FilterShader = new PhysXFilterShader()
            };

            Scene = Physics.CreateScene(sceneDesc);
            ControllerManager = Scene.CreateControllerManager();

            // Default physics material: staticFriction, dynamicFriction, restitution
            DefaultMaterial = Physics.CreateMaterial(0.7f, 0.7f, 0.1f);

            Debug.Log("[PhysX] Initialized — gravity (0, -9.81, 0)");
        }

        /// <summary>
        /// Make every character controller forget the shapes it remembers round it. Call after static geometry was added
        /// or taken away in play (the water's ice, WaterBody): a controller keeps what it found in a volume round it and
        /// walks through a shape made since until it leaves that volume (2026-10-23: ice turned off and on again).
        /// </summary>
        public static void InvalidateControllerCaches()
        {
            var manager = ControllerManager;
            if (manager == null) return;
            // The list keeps controllers whose wrapper was disposed (the project's characters dispose their capsule's actor
            // first, then the controller): InvalidateCache on one threw, and the first water body to swap its ice after a
            // step took the blame (2026-10-24: 'Stillwater' in the player, a fountain in the editor)
            foreach (var controller in manager.Controllers)
                if (controller is { Disposed: false })
                    controller.InvalidateCache();
        }

        // The PhysX Visual Debugger's default port
        private const int PvdPort = 5425;

        private static bool IsListening(int port)
        {
            try
            {
                foreach (var endpoint in System.Net.NetworkInformation.IPGlobalProperties.GetIPGlobalProperties().GetActiveTcpListeners())
                    if (endpoint.Port == port) return true;
            }
            catch (System.Net.NetworkInformation.NetworkInformationException) { }
            return false;
        }

        /// <summary>Steps the simulation has taken: what moved in the last one changes when this does.</summary>
        public static long Steps { get; private set; }

        /// <summary>
        /// How far the picture is into the step after the last one, 0..1: what moves by the simulation is drawn that
        /// share of the way from where the step before the last left it to where the last did (Ragdoll), so that it
        /// moves smoothly at more frames a second than the simulation's 60.
        /// </summary>
        public static float StepFraction => Math.Clamp(_accumulator / FixedTimestep, 0f, 1f);

        /// <summary>Seconds a step covers.</summary>
        public static float StepSeconds => FixedTimestep;

        public static void Update(float deltaTime)
        {
            _accumulator += deltaTime;
            if (_accumulator < FixedTimestep)
                return;

            _accumulator -= FixedTimestep;
            Scene.Simulate(FixedTimestep);
            Scene.FetchResults(true);
            Steps++;

            foreach (var body in _simulated)
            {
                if (body.Actor is RigidDynamic dynamic && dynamic.UserData is Entity entity)
                {
                    entity.Transform.Position = dynamic.GlobalPosePosition;
                    entity.Transform.Rotation = dynamic.GlobalPoseQuat;
                }
            }
        }

        // The bodies the simulation moves (a RigidBody that is neither static nor kinematic): after a step their
        // entities are put where their actors went. Kept by RigidBody, so a step walks only these: Scene.GetActors
        // walked every dynamic actor (the characters' controllers, the animals' capsules, every door's leaf), and
        // would have put a kinematic body back where its actor was, when it is the actor that follows the transform.
        private static readonly HashSet<Components.RigidBody> _simulated = new();

        internal static void Simulated(Components.RigidBody body, bool simulated)
        {
            if (simulated) _simulated.Add(body);
            else _simulated.Remove(body);
        }

        public static void Flush()
        {
            Scene.FlushSimulation(false);
        }

        /// <summary>
        /// Performs a raycast against the physics scene, returning the closest hit that is not a trigger.
        /// A ray that starts inside a shape hits it at the origin, with the normal facing back along the ray.
        /// </summary>
        public static bool Raycast(Vector3 origin, Vector3 direction, float maxDistance,
                                   out Vector3 hitPosition, out Vector3 hitNormal)
        {
            hitPosition = origin;
            hitNormal = Vector3.UnitY;

            if (Scene == null) return false;

            direction = Vector3.Normalize(direction);

            Vector3 nrm = Vector3.UnitY;
            float closestDist = float.MaxValue;
            bool found = false;

            // The query hands over every hit along the ray, up to maximumHits per call and in no
            // order, so the closest has to be picked out here. hit.Normal is only filled when the
            // flag asks for it; the position comes from the distance, which is always right.
            Scene.Raycast(origin, direction, maxDistance, 8,
                (hits) =>
                {
                    if (hits == null) return true;

                    foreach (var hit in hits)
                    {
                        if (hit == null || hit.Distance >= closestDist) continue;
                        if (hit.Shape != null && (hit.Shape.Flags & ShapeFlag.TriggerShape) != 0) continue;

                        closestDist = hit.Distance;
                        nrm = hit.Normal;
                        found = true;
                    }
                    return true;
                }, HitFlag.Normal);

            if (found)
            {
                hitPosition = origin + direction * closestDist;
                hitNormal = nrm;
            }

            return found;
        }

        public static void Shutdown()
        {
            Scene?.Dispose();
            Physics?.Dispose();
            Debug.Log("[PhysX] Shutdown");
        }
    }
}
