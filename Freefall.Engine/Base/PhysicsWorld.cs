using System;
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
        public override FilterResult Filter(int attributes0, FilterData filterData0, int attributes1, FilterData filterData1)
        {
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

        public static void Update(float deltaTime)
        {
            _accumulator += deltaTime;
            if (_accumulator < FixedTimestep)
                return;

            _accumulator -= FixedTimestep;
            Scene.Simulate(FixedTimestep);
            Scene.FetchResults(true);

            var activeActors = Scene.GetActors(ActorTypeFlag.RigidDynamic);
            foreach (var actor in activeActors)
            {
                if (actor is RigidDynamic dynamic && actor.UserData is Entity entity)
                {
                    entity.Transform.Position = dynamic.GlobalPosePosition;
                    entity.Transform.Rotation = dynamic.GlobalPoseQuat;
                }
            }
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
