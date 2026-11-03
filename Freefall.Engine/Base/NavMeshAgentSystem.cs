using System;
using Freefall.Components;

namespace Freefall.Base
{
    /// <summary>
    /// The agents' frame, in two parts:
    ///
    /// - **The path service** (<see cref="NavMeshWorld.ServeWays"/>): the ways the agents' destinations asked for, a budget
    ///   of time a frame for all of them together (<see cref="NavMeshWorld.WayMilliseconds"/>), in the order they asked. A
    ///   search that takes more goes on in the next frame. Each way found becomes its agent's crowd target and Way.
    /// - **Every <see cref="NavMeshAgent"/>'s update** (<see cref="NavMeshAgent.Update"/>): an agent the crowd drives takes
    ///   the crowd's step (taken in <c>Engine.Tick</c> before the update group), and a body a character moved is put back
    ///   into the crowd for its next step.
    ///
    /// Play only, as the agent's update always was. In <see cref="ScriptGroup"/>, ahead of the scripts' IUpdate dispatch.
    /// A game's character system orders itself before this one, so the agents take what the characters did this frame,
    /// and the destinations its brains set this frame are searched for at once (systems over IUpdate, 2026-11-03).
    /// </summary>
    [UpdateInGroup(typeof(ScriptGroup))]
    [UpdateBefore(typeof(ScriptUpdateSystem))]
    public sealed class NavMeshAgentSystem : EntitySystem
    {
        public override void Update()
        {
            NavMeshWorld.ServeWays();

            // By index: an agent's update may add or remove agents (a reseat never does, a script reacting might)
            var agents = ComponentCache<NavMeshAgent>.All;
            for (int i = 0; i < agents.Count; i++)
            {
                var agent = agents[i];
                if (agent == null || !agent.Enabled || agent.IsDestroyed) continue;
                try { agent.Update(); }
                catch (Exception e) { CallbackErrors.Report(agent, nameof(NavMeshAgent.Update), e); }
            }
        }
    }
}
