using System.Globalization;
using Freefall.Assets;
using Freefall.Assets.Importers;

namespace ReimportTest
{
    /// <summary>
    /// Offline test of the forced reimport (AssetDatabase.GetPathsInFolder / ReimportFolder / ReimportFiles /
    /// TickReimports) and of importer settings surviving it, without the editor: builds a throwaway project of
    /// small OBJ files (and one broken FBX) in the temp folder, imports it, and drives the reimport with
    /// TickReimports as Engine.Tick would. Prints every check; exit code 0 when all pass.
    ///
    ///   ReimportTest [--keep]      --keep leaves the temp project on disk
    /// </summary>
    static class Program
    {
        static int _failures;

        static int Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            bool keep = args.Contains("--keep");
            Out = Console.Out;
            Console.SetOut(TextWriter.Null);

            var root = Path.Combine(Path.GetTempPath(), "FreefallReimportTest_" + Guid.NewGuid().ToString("N")[..8]);
            try
            {
                Run(root);
            }
            catch (Exception ex)
            {
                Check(false, $"no exception ({ex.GetType().Name}: {ex.Message})");
                Out.WriteLine(ex);
            }
            finally
            {
                if (!keep)
                    try { Directory.Delete(root, recursive: true); } catch { }
                else
                    Out.WriteLine($"Project kept at {root}");
            }

            if (_failures > 0)
            {
                Out.WriteLine();
                Out.WriteLine("Log:");
                foreach (var line in Freefall.Debug.Lines)
                    Out.WriteLine("  " + line.Message);
            }

            Out.WriteLine(_failures == 0 ? "ALL PASSED" : $"{_failures} FAILED");
            return _failures == 0 ? 0 : 1;
        }

        // The checks' output; Console itself is muted because Debug writes every import line to it
        static TextWriter Out;

        static void Run(string root)
        {
            var project = FreefallProject.Create(root, "ReimportTest");
            var assets = project.AssetsDirectory;

            WriteCube(Path.Combine(assets, "Pack", "a.obj"));
            WriteCube(Path.Combine(assets, "Pack", "Sub", "b.obj"));
            WriteCube(Path.Combine(assets, "PackB", "c.obj"));                        // shares the prefix "Pack", not the folder
            Directory.CreateDirectory(Path.Combine(assets, "Pack"));
            File.WriteAllText(Path.Combine(assets, "Pack", "broken.fbx"), "not an fbx"); // its import fails

            AssetDatabase.Initialize(project);
            AssetDatabase.ImportAll();

            string a = @"Pack\a.obj", b = @"Pack\Sub\b.obj", c = @"PackB\c.obj", broken = @"Pack\broken.fbx";
            foreach (var path in new[] { a, b, c })
                Check(AssetDatabase.GetMeta(AssetDatabase.PathToGuid(path))?.SubAssets.Count > 0, $"{path} imported at the start");

            // ── Which files a folder means ──
            Expect(AssetDatabase.GetPathsInFolder("Pack"), new[] { a, broken, b }, "folder 'Pack' (sub-folders in, 'PackB' out)");
            Expect(AssetDatabase.GetPathsInFolder("Pack/"), new[] { a, broken, b }, "folder 'Pack/'");
            Expect(AssetDatabase.GetPathsInFolder(@"\Pack\"), new[] { a, broken, b }, @"folder '\Pack\'");
            Expect(AssetDatabase.GetPathsInFolder(Path.Combine(assets, "Pack")), new[] { a, broken, b }, "absolute folder");
            Expect(AssetDatabase.GetPathsInFolder("Pack", new[] { "obj" }), new[] { a, b }, "extension 'obj'");
            Expect(AssetDatabase.GetPathsInFolder("Pack", new[] { ".OBJ" }), new[] { a, b }, "extension '.OBJ'");
            Expect(AssetDatabase.GetPathsInFolder("Pack", new[] { "png" }), Array.Empty<string>(), "extension 'png' (none)");
            Expect(AssetDatabase.GetPathsInFolder("Pac"), Array.Empty<string>(), "'Pac' is no folder");
            Expect(AssetDatabase.GetPathsInFolder("pack/a.obj"), new[] { a }, "a file path gives that file");
            Check(AssetDatabase.GetPathsInFolder("").Count == 4, "'' = all of Assets (4 files)");

            // ── A forced reimport of unchanged files ──
            var before = Stamps(a, b, c, broken);
            Thread.Sleep(30);

            var job = AssetDatabase.ReimportFolder("Pack", new[] { "obj" });
            int completedEvents = 0;
            job.Completed += _ => completedEvents++;
            Check(job.Total == 2 && job.Label == "Pack", $"job: 2 files, label 'Pack' (got {job.Total}, '{job.Label}')");
            Check(AssetDatabase.ActiveReimport == job, "job is the active reimport");
            Check(Throws(() => AssetDatabase.ReimportFolder("PackB")), "a second reimport while one runs throws");

            int ticks = Drive(job);
            var after = Stamps(a, b, c, broken);
            Check(job.IsCompleted && !job.Aborted, $"job completed after {ticks} ticks");
            Check(job.Done == 2 && job.Imported == 2 && job.Failed == 0, $"2 imported, 0 failed (got {job.Imported}, {job.Failed})");
            Check(after[a] > before[a] && after[b] > before[b], "a.obj and Sub/b.obj imported again although unchanged");
            Check(after[c] == before[c], "PackB/c.obj left alone");
            Check(after[broken] == before[broken], "broken.fbx left alone (extension filter)");
            Check(completedEvents == 1, "Completed raised once");
            Check(AssetDatabase.ActiveReimport == null, "no active reimport afterwards");

            // ── A failing file is counted, the rest go on ──
            var all = AssetDatabase.ReimportFolder("Pack");
            Drive(all);
            Check(all.Total == 3 && all.Imported == 2 && all.Failed == 1, $"whole folder: 2 imported, 1 failed (got {all.Imported}, {all.Failed})");
            Check(all.FailedPaths.SequenceEqual(new[] { broken }), $"failed path is broken.fbx (got {string.Join(", ", all.FailedPaths)})");

            // ── Importer settings: saved, then kept through a forced reimport ──
            var guid = AssetDatabase.PathToGuid(a);
            var importer = AssetDatabase.GetImporter(guid) as ModelImporter;
            Check(importer != null, "a.obj has a ModelImporter");
            Check(AssetDatabase.GetImporterSourceGuid(importer) == guid, "GetImporterSourceGuid finds the importer's file");
            Check(AssetDatabase.GetImporterSourceGuid(new ModelImporter()) == null, "an importer made elsewhere has no file");

            var names = AssetDatabase.GetImporterSettingNames(importer);
            Check(ModelImporter.LODGenerationMembers.All(names.Contains), "setting names include the generated-LOD settings");
            Check(!names.Contains(nameof(ModelImporter.OverrideSettings)) && !names.Contains(nameof(ModelImporter.LODGenerationMembers)),
                "setting names leave out static members");
            var texture = AssetDatabase.GetImporterSettingNames(new TextureImporter());
            Check(texture.Contains(nameof(TextureImporter.Format)) && texture.Contains(nameof(TextureImporter.UserConfigured))
                  && !texture.Contains("AssetType"), "TextureImporter setting names: Format, UserConfigured, no AssetType");

            importer.LODPixelError = 4;
            importer.LODRatios = new List<float> { 0.6f, 0.3f };
            AssetDatabase.SaveImporter(guid, importer);
            var one = AssetDatabase.ReimportFiles(new[] { a }, a);
            Drive(one);
            var reread = AssetDatabase.GetImporter(guid) as ModelImporter;
            Check(one.Imported == 1, "the file with new settings reimported");
            Check(reread.LODPixelError == 4 && reread.LODRatios.SequenceEqual(new[] { 0.6f, 0.3f }),
                $"settings kept through the reimport (LODPixelError {reread.LODPixelError}, LODRatios {string.Join("/", reread.LODRatios)})");

            // ── A project change ends a running reimport ──
            var stopped = AssetDatabase.ReimportFolder("Pack");
            AssetDatabase.Initialize(project);
            Check(stopped.IsCompleted && stopped.Aborted, "opening a project aborts the running reimport");
            Check(AssetDatabase.ActiveReimport == null, "no active reimport after the project change");
        }

        /// <summary>Tick until the job completes, as Engine.Tick would once per frame.</summary>
        static int Drive(ReimportJob job)
        {
            int ticks = 0;
            while (!job.IsCompleted && ticks < 1000)
            {
                AssetDatabase.TickReimports();
                ticks++;
            }
            return ticks;
        }

        static Dictionary<string, DateTime> Stamps(params string[] paths) =>
            paths.ToDictionary(p => p, p => AssetDatabase.GetMeta(AssetDatabase.PathToGuid(p)).LastImported);

        static bool Throws(Action action)
        {
            try { action(); return false; }
            catch (InvalidOperationException) { return true; }
        }

        static void Expect(List<string> actual, string[] expected, string what) =>
            Check(actual.SequenceEqual(expected, StringComparer.OrdinalIgnoreCase),
                $"{what}: [{string.Join(", ", expected)}]" + (actual.SequenceEqual(expected, StringComparer.OrdinalIgnoreCase) ? "" : $" — got [{string.Join(", ", actual)}]"));

        static void Check(bool ok, string what)
        {
            Out.WriteLine($"{(ok ? "PASS" : "FAIL")}  {what}");
            if (!ok) _failures++;
        }

        /// <summary>A unit cube as OBJ: 8 vertices, 12 triangles.</summary>
        static void WriteCube(string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, """
                o Cube
                v -0.5 -0.5 -0.5
                v  0.5 -0.5 -0.5
                v  0.5  0.5 -0.5
                v -0.5  0.5 -0.5
                v -0.5 -0.5  0.5
                v  0.5 -0.5  0.5
                v  0.5  0.5  0.5
                v -0.5  0.5  0.5
                f 1 3 2
                f 1 4 3
                f 5 6 7
                f 5 7 8
                f 1 2 6
                f 1 6 5
                f 4 7 3
                f 4 8 7
                f 1 5 8
                f 1 8 4
                f 2 3 7
                f 2 7 6
                """);
        }
    }
}
