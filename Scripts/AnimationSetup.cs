using System;
using System.Numerics;
using Freefall.Animation;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.Scripts
{
    /// <summary>
    /// Sets up animations for the Player character using the Paladin/Knight models.
    /// Matches Apex's SceneUtil.CreateBlendTreeAnimations pattern.
    /// </summary>
    public static class AnimationSetup
    {
        public static void CreateBlendTreeAnimations(Entity entity, Mesh mesh)
        {
            string animPath = @"D:\Projects\2024\ProjectXYZ\Resources\Characters\Knight\";

            // Load animation clips
            var walkAnim = Engine.Assets.Load<AnimationClip>($"{animPath}Walking.dae-anim");
            var idleAnim = Engine.Assets.Load<AnimationClip>($"{animPath}Idle.dae-anim");
            var runAnim = Engine.Assets.Load<AnimationClip>($"{animPath}Running.dae-anim");
            var walkBack = Engine.Assets.Load<AnimationClip>($"{animPath}Walking Backward.dae-anim");
            var strafeRightAnim = Engine.Assets.Load<AnimationClip>($"{animPath}Right Strafe Walking.dae-anim");
            var strafeLeftAnim = Engine.Assets.Load<AnimationClip>($"{animPath}Left Strafe Walking.dae-anim");
            var strafeRightRunAnim = Engine.Assets.Load<AnimationClip>($"{animPath}Right Strafe.dae-anim");
            var strafeLeftRunAnim = Engine.Assets.Load<AnimationClip>($"{animPath}Left Strafe.dae-anim");
            var jogBack = Engine.Assets.Load<AnimationClip>($"{animPath}Jog Backward.dae-anim");
            var jumpAnim = Engine.Assets.Load<AnimationClip>($"{animPath}Jumping Up Quick.dae-anim");
            var falling = Engine.Assets.Load<AnimationClip>($"{animPath}Falling Idle.dae-anim");
            var landing = Engine.Assets.Load<AnimationClip>($"{animPath}Jumping Down Quick.dae-anim");

            // Create animation states
            var idleState = new AnimationState { Name = "Idle", Clip = idleAnim, Loop = true };
            var walkState = new AnimationState { Name = "Walk", Clip = walkAnim, Loop = true };
            var walkBackState = new AnimationState { Name = "WalkBack", Clip = walkBack, Loop = true };
            var runState = new AnimationState { Name = "Run", Clip = runAnim, Loop = true };
            var strafeRightState = new AnimationState { Name = "StrafeRight", Clip = strafeRightAnim, Loop = true };
            var strafeLeftState = new AnimationState { Name = "StrafeLeft", Clip = strafeLeftAnim, Loop = true };
            var strafeRightRunState = new AnimationState { Name = "StrafeRightRun", Clip = strafeRightRunAnim, Loop = true };
            var strafeLeftRunState = new AnimationState { Name = "StrafeLeftRun", Clip = strafeLeftRunAnim, Loop = true };
            var jogBackState = new AnimationState { Name = "JogBack", Clip = jogBack, Loop = true };
            var jumpState = new AnimationState { Name = "Jump", Clip = jumpAnim, Loop = false };
            var fallingState = new AnimationState { Name = "Falling", Clip = falling, Loop = true };
            var landingState = new AnimationState { Name = "Landing", Clip = landing, Loop = false };

            // Add animation events
            walkAnim.Events.Add(new AnimationEvent { Name = "footstep", Time = 0.3f });
            walkAnim.Events.Add(new AnimationEvent { Name = "footstep", Time = 0.75f });
            runAnim.Events.Add(new AnimationEvent { Name = "footstep", Time = 0.3f });
            runAnim.Events.Add(new AnimationEvent { Name = "footstep", Time = 0.75f });
            jumpAnim.Events.Add(new AnimationEvent { Name = "jump", Time = 0.001f });
            landing.Events.Add(new AnimationEvent { Name = "land", Time = 0.001f });

            // Create blend tree for locomotion
            var blendTree = new AnimationBlendTree
            {
                Name = "Locomotion",
                ParameterA = "axisX",
                ParameterB = "axisY",
            };

            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = idleState, Values = new Vector2(0, 0) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = walkState, Values = new Vector2(0, 0.5f) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = runState, Values = new Vector2(0, 1f) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = walkBackState, Values = new Vector2(0, -.5f) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = jogBackState, Values = new Vector2(0, -1f) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = strafeRightState, Values = new Vector2(.5f, 0) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = strafeLeftState, Values = new Vector2(-.5f, 0) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = strafeRightRunState, Values = new Vector2(1f, 0) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = strafeLeftRunState, Values = new Vector2(-1f, 0) });

            // Create transitions
            var toJump = new AnimationTransition
            {
                Source = blendTree,
                Target = jumpState,
                Conditions = {
                    new AnimationCondition {
                        Parameter = "jump",
                        Comparison = ComparisonType.Greater,
                        Value = 0
                    }
                }
            };

            var toFalling = new AnimationTransition
            {
                Source = jumpState,
                Target = fallingState,
            };

            var toLanding = new AnimationTransition
            {
                Source = fallingState,
                Target = landingState,
                Conditions = {
                    new AnimationCondition {
                        Parameter = "landing",
                        Comparison = ComparisonType.Greater,
                        Value = 0
                    }
                }
            };

            var toBlendTree = new AnimationTransition
            {
                Source = landingState,
                Target = blendTree,
            };

            var fromBlendToFalling = new AnimationTransition
            {
                Source = blendTree,
                Target = fallingState,
                Conditions = {
                    new AnimationCondition {
                        Parameter = "falling",
                        Comparison = ComparisonType.Greater,
                        Value = 0
                    }
                }
            };

            // Create animation layer
            var layer = new AnimationLayer
            {
                Name = "Default",
                States = { blendTree, jumpState, fallingState, landingState },
                Transitions = { toJump, toFalling, toLanding, toBlendTree, fromBlendToFalling }
            };

            // Create animation controller
            var animation = new Animation.Animation
            {
                Name = "PlayerAnimation"
            };
            animation.Layers.Add(layer);
            animation.Parameters.Add("axisX", 0);
            animation.Parameters.Add("axisY", 0);
            animation.Parameters.Add("jump", 0);
            animation.Parameters.Add("landing", 0);
            animation.Parameters.Add("falling", 0);

            // Add animator component
            var animator = entity.AddComponent<Animator>();
            animator.Animation = animation;

            Debug.Log($"[AnimationSetup] Created animation controller with {animation.Layers.Count} layers");
        }
    }
}
