using System;
using System.Numerics;
using PhysX;

namespace Freefall.Base
{
    public class PhysXErrorOutput : ErrorCallback
    {
        public override void ReportError(ErrorCode errorCode, string message, string file, int lineNumber)
        {
            Debug.Log($"[PhysX] {errorCode}: {message}");
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
        public static PhysX.Material DefaultMaterial { get; private set; }

        public static void Initialize()
        {
            var errorOutput = new PhysXErrorOutput();
            var foundation = new Foundation(errorOutput);

            // PVD — attempt connection, fails gracefully if PVD app is not running
            var pvd = new PhysX.VisualDebugger.Pvd(foundation);
            Physics = new PhysX.Physics(foundation, true, pvd);
            
            try
            {
                Physics.Pvd.Connect("localhost");
                Debug.Log("[PhysX] PVD connected");
            }
            catch
            {
                Debug.Log("[PhysX] PVD not available (optional)");
            }

            // Create scene with gravity and collision filtering
            var sceneDesc = new SceneDesc
            {
                Gravity = new Vector3(0, -9.81f, 0),
                FilterShader = new PhysXFilterShader()
            };

            Scene = Physics.CreateScene(sceneDesc);

            // Default physics material: staticFriction, dynamicFriction, restitution
            DefaultMaterial = Physics.CreateMaterial(0.7f, 0.7f, 0.1f);

            Debug.Log("[PhysX] Initialized — gravity (0, -9.81, 0)");
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
            foreach (RigidDynamic actor in activeActors) 
            {
                var entity = actor.UserData as Entity;
                if (entity != null)
                {
                    entity.Transform.Position = actor.GlobalPosePosition;
                    entity.Transform.Rotation = actor.GlobalPoseQuat;
                }
            }
        }

        /// <summary>
        /// Performs a raycast against the physics scene, returning the closest hit.
        /// </summary>
        public static bool Raycast(Vector3 origin, Vector3 direction, float maxDistance,
                                   out Vector3 hitPosition, out Vector3 hitNormal)
        {
            hitPosition = origin;
            hitNormal = Vector3.UnitY;

            if (Scene == null) return false;

            Vector3 pos = origin;
            Vector3 nrm = Vector3.UnitY;
            float closestDist = maxDistance;
            bool found = false;

            Scene.Raycast(origin, Vector3.Normalize(direction), maxDistance, 1,
                (hits) =>
                {
                    if (hits != null && hits.Length > 0)
                    {
                        var hit = hits[0];
                        pos = hit.Position;
                        nrm = hit.Normal;
                        found = true;
                    }
                    return true;
                });

            if (found)
            {
                hitPosition = pos;
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
