using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;
using Squid;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using IA = Freefall.Assets.InternalAssets;
using Reflector = Freefall.Reflection.Reflector;

namespace Freefall.Editor
{
    /// <summary>
    /// Data carried during asset drag & drop operations.
    /// </summary>
    public class AssetDragData
    {
        public string Guid;
        public Type AssetType;

        private static Dictionary<string, Type> _typeMap;

        public static Type ResolveType(string typeName)
        {
            if (string.IsNullOrEmpty(typeName)) return typeof(Asset);

            _typeMap ??= BuildTypeMap();

            if (_typeMap.TryGetValue(typeName, out var t))
                return t;

            return typeof(Asset);
        }

        /// <summary>Forget the map after a script reload: it holds script asset types (and so the old script assembly).</summary>
        internal static void ResetTypeMap() => _typeMap = null;

        private static Dictionary<string, Type> BuildTypeMap()
        {
            var map = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
            var assetType = typeof(Asset);

            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                if (ScriptCompiler.IsStale(asm)) continue;
                try
                {
                    foreach (var type in asm.GetTypes())
                    {
                        if (type.IsAbstract || !assetType.IsAssignableFrom(type)) continue;
                        map.TryAdd(type.Name, type);

                        // Also register aliases declared via [AssetTypeAlias]
                        foreach (var alias in type.GetCustomAttributes(typeof(AssetTypeAliasAttribute), false))
                        {
                            var attr = (AssetTypeAliasAttribute)alias;
                            if (!string.IsNullOrEmpty(attr.Alias))
                                map.TryAdd(attr.Alias, type);
                        }
                    }
                }
                catch { /* ignore reflection errors from unloadable assemblies */ }
            }

            return map;
        }
    }

    /// <summary>
    /// Lightweight data for a single card in the asset grid.
    /// </summary>
    public struct CardData
    {
        public string Name;
        public string TypeName;
        public string SourceGuid;
        public string SubGuid;
        public string SubType;
        public Type DragAssetType;
        public bool IsFolder;
        public DirectoryInfo FolderInfo;
        public bool IsSourceCard;
        /// <summary>Absolute path for non-asset files (e.g. .cs scripts) that aren't in the asset DB.</summary>
        public string FullPath;
    }

    /// <summary>
    /// Asset browser panel. Left: folder tree (VirtualList). Right: virtualized content grid.
    /// Driven by AssetDatabase's GUID/meta system — only shows importable files.
    /// The content grid is a VirtualList where each item is a row of cards.
    /// </summary>
    public class AssetBrowserControl : Frame
    {
        // ── Left panel: folder tree ──
        private VirtualList _folderList;
        private readonly List<FolderNode> _folders = new();

        // ── Right panel: virtualized content grid ──
        private SearchBox _searchBox;
        private Frame _breadcrumb;
        private VirtualList _contentList;
        private readonly List<CardData> _dataSource = new();
        private readonly Stack<Button> _cardPool = new();

        // ── Grid layout state ──
        private int _columns = 1;
        private int _cardWidth = 120;
        private int _cardExtra = 60;
        private const int CardSpacing = 8;
        private const int MinCardWidth = 64;
        private const int MaxCardWidth = 256;
        private Slider _sizeSlider;

        // Tracks which folder is selected
        private FolderNode _selectedFolder;

        // ── Selected card tracking for rename ──
        private Button _selectedCard;
        private string _selectedCardGuid;

        // ── Inline rename state ──
        private TextBox _renameBox;
        private Button _renamingCard;
        private string _renamingGuid;
        private DropDownButton btnCreate;

        // ── Persistent expand state ──
        private static readonly HashSet<string> _expandedPaths = new();

        private int CardHeight => _cardWidth + _cardExtra;
        private int RowCount => _columns > 0 ? (int)Math.Ceiling((float)_dataSource.Count / _columns) : 0;

        public AssetBrowserControl()
        {
            Style = "";
            Dock = DockStyle.Fill;

            var split = new SplitContainer
            {
                Dock = DockStyle.Fill,
                RetainAspect = false,
                Orientation = Orientation.Horizontal,
            };
            split.SplitFrame1.Size = new Point(200, 200);
            split.SplitButton.Size = new Point(2, 2);
            split.SplitButton.Style = "frame";
            split.SplitButton.Margin = new Margin(1, 0, 1, 0);

            // ── Left: folder tree ──
            var leftToolbar = new Frame
            {
                Style = "header",
                Size = new Point(16, 26),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 0, 0, 1)
            };

            var folderLabel = new Label
            {
                Text = "Folders",
                Dock = DockStyle.Fill,
                Margin = new Margin(6, 0, 0, 0)
            };
            leftToolbar.Controls.Add(folderLabel);

            _folderList = new VirtualList
            {
                Dock = DockStyle.Fill,
                ItemHeight = 20
            };
            _folderList.Scrollbar.ButtonDown.Visible = false;
            _folderList.Scrollbar.ButtonUp.Visible = false;
            _folderList.Scrollbar.Slider.Ease = false;
            _folderList.Scrollbar.Slider.MinHandleSize = 64;
            _folderList.Content.Style = "";
            _folderList.CreateItem = CreateFolderItem;
            _folderList.BindItem = BindFolderItem;

            split.SplitFrame1.Controls.Add(leftToolbar);
            split.SplitFrame1.Controls.Add(_folderList);

            // ── Right: toolbar + content ──
            var rightToolbar = new Frame
            {
                Style = "header",
                Size = new Point(16, 26),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 0, 0, 1)
            };

            _searchBox = new SearchBox
            {
                Size = new Point(200, 16),
                Dock = DockStyle.Left,
                Margin = new Margin(2)
            };
            _searchBox.TextChanged += OnSearchChanged;

            _breadcrumb = new Frame
            {
                Dock = DockStyle.Left,
                AutoSize = AutoSize.Horizontal,
                Margin = new Margin(8, 0, 0, 0)
            };

            // Create asset dropdown button
            btnCreate = new DropDownButton
            {
                Text = "Create",
                Style = "button",
                Size = new Point(70, 20),
                Margin = new Margin(0, 2, 4, 2),
                Dock = DockStyle.Right,
            };

            var types = Reflector.GetTypes<Asset>();
            var creatables = new List<(string Label, Type Type)>();
            foreach (var type in types)
            {
                var attr = Reflector.GetAttribute<CreateAssetAttribute>(type);
                if (attr != null)
                {
                    var label = string.IsNullOrEmpty(attr.Caption) ? type.Name : attr.Caption;
                    creatables.Add((label, type));
                }
            }

            int itemHeight = 26;
            int dropWidth = 160;
            int dropHeight = creatables.Count * (itemHeight + 1) + 4;

            btnCreate.Dropdown.Style = "window";
            btnCreate.Dropdown.Padding = new Margin(2);
            btnCreate.Dropdown.Size = new Point(dropWidth, dropHeight);
            btnCreate.Dropdown.Resizable = false;

            foreach (var (label, assetType) in creatables)
            {
                var item = new Button
                {
                    Text = label,
                    Style = "item",
                    Size = new Point(dropWidth - 8, itemHeight),
                    Dock = DockStyle.Top,
                    Margin = new Margin(0, 0, 0, 1),
                    Tag = assetType,
                };
                item.MouseClick += (s, e) =>
                {
                    if (e.Button > 0) return;
                    var type = (Type)s.Tag;
                    var folderPath = _selectedFolder?.Info.FullName;
                    if (string.IsNullOrEmpty(folderPath)) return;
                    var guid = AssetCreator.CreateAsset(type, folderPath);
                    MessageDispatcher.Send(Msg.RefreshAssets);

                    // Auto-enter rename mode on the new card
                    if (!string.IsNullOrEmpty(guid))
                        _pendingRenameGuid = guid;
                };
                btnCreate.Dropdown.Controls.Add(item);
            }

            rightToolbar.Controls.Add(btnCreate);
            rightToolbar.Controls.Add(_searchBox);
            rightToolbar.Controls.Add(_breadcrumb);

            // Bottom bar with card size slider
            var bottomBar = new Frame
            {
                Style = "frame",
                Size = new Point(16, 20),
                Dock = DockStyle.Bottom,
                Margin = new Margin(0, 1, 0, 0)
            };

            _sizeSlider = new Slider
            {
                Orientation = Orientation.Horizontal,
                Size = new Point(100, 14),
                Dock = DockStyle.Right,
                Margin = new Margin(4, 3, 4, 3),
                Minimum = MinCardWidth,
                Maximum = MaxCardWidth,
                Ease = false,
                Style = "sliderTrack",
            };
            _sizeSlider.Button.Style = "sliderThumb";
            _sizeSlider.Button.Size = new Point(16, 14);
            _sizeSlider.SetValue(_cardWidth);
            _sizeSlider.ValueChanged += (s) =>
            {
                _cardWidth = (int)_sizeSlider.Value;
                UpdateColumns();
            };
            bottomBar.Controls.Add(_sizeSlider);

            _contentList = new VirtualList
            {
                Dock = DockStyle.Fill,
                ItemHeight = CardHeight + CardSpacing,
                Style = "frame"
            };
            _contentList.Scrollbar.ButtonDown.Visible = false;
            _contentList.Scrollbar.ButtonUp.Visible = false;
            _contentList.Scrollbar.Slider.Ease = false;
            _contentList.Scrollbar.Slider.MinHandleSize = 64;
            _contentList.Content.Style = "";
            _contentList.CreateItem = CreateRow;
            _contentList.BindItem = BindRow;

            _contentList.AllowDrop = true;
            _contentList.DragDrop += (s, e) =>
            {
                if (e.DraggedControl?.Tag is not Entity entity)
                    return;

                var folderPath = _selectedFolder?.Info.FullName;
                var prefab = Prefab.Create(entity);
                string ext = AssetManager.GetFileExtension(typeof(Prefab));
                string savePath = Path.Combine(folderPath, prefab.Name + ext);

                try
                {
                    Engine.Assets.SaveAsset(prefab, savePath);
                    MessageDispatcher.Send(Msg.RefreshAssets);

                    Debug.Log($"[AssetCreator] Created {prefab.Name}: {savePath}");
                }
                catch (Exception ex)
                {
                    Debug.Log($"[AssetCreator] Failed to create {prefab.Name}: {ex.Message}");
                }
            };

            split.SplitFrame2.Controls.Add(rightToolbar);
            split.SplitFrame2.Controls.Add(bottomBar);
            split.SplitFrame2.Controls.Add(_contentList);

            Controls.Add(split);

            MessageDispatcher.AddListener(Msg.ScriptsReloaded, (msg) => BuildCreateMenu());
        }

        /// <summary>
        /// Initialize the browser after AssetDatabase is ready.
        /// Call this once the project is loaded (after AssetDatabase.Initialize).
        /// </summary>
        public void Initialize()
        {
            var project = AssetDatabase.Project;
            if (project == null) return;

            var assetsDir = project.AssetsDirectory;
            if (!Directory.Exists(assetsDir)) return;

            RebuildFolderTree(assetsDir);

            _folderList.DataSource = _folders;

            // Select root by default
            SelectFolder(_folders[0]);
        }

        private void BuildCreateMenu()
        {
            btnCreate.Dropdown.Controls.Clear();

            var types = Reflector.GetTypes<Asset>();
            var creatables = new List<(string Label, Type Type)>();
            foreach (var type in types)
            {
                var attr = Reflector.GetAttribute<CreateAssetAttribute>(type);
                if (attr != null)
                {
                    var label = string.IsNullOrEmpty(attr.Caption) ? type.Name : attr.Caption;
                    creatables.Add((label, type));
                }
            }

            int itemHeight = 26;
            int dropWidth = 160;
            int dropHeight = creatables.Count * (itemHeight + 1) + 4;

            btnCreate.Dropdown.Style = "window";
            btnCreate.Dropdown.Padding = new Margin(2);
            btnCreate.Dropdown.Size = new Point(dropWidth, dropHeight);
            btnCreate.Dropdown.Resizable = false;

            foreach (var (label, assetType) in creatables)
            {
                var item = new Button
                {
                    Text = label,
                    Style = "item",
                    Size = new Point(dropWidth - 8, itemHeight),
                    Dock = DockStyle.Top,
                    Margin = new Margin(0, 0, 0, 1),
                    Tag = assetType,
                };
                item.MouseClick += (s, e) =>
                {
                    if (e.Button > 0) return;
                    var type = (Type)s.Tag;
                    var folderPath = _selectedFolder?.Info.FullName;
                    if (string.IsNullOrEmpty(folderPath)) return;
                    var guid = AssetCreator.CreateAsset(type, folderPath);
                    MessageDispatcher.Send(Msg.RefreshAssets);

                    // Auto-enter rename mode on the new card
                    if (!string.IsNullOrEmpty(guid))
                        _pendingRenameGuid = guid;
                };
                btnCreate.Dropdown.Controls.Add(item);
            }
        }

        /// <summary>
        /// Rebuild the folder tree from the assets directory.
        /// Always appends the virtual InternalAssets node at the end.
        /// </summary>
        private void RebuildFolderTree(string assetsDir)
        {
            _folders.Clear();
            var root = new FolderNode(null, new DirectoryInfo(assetsDir), 0);
            root.Expanded = true;
            _folders.Add(root);
            GatherExpandedChildren(root, _folders);

            // Virtual "InternalAssets" folder (not filesystem-backed)
            _folders.Add(new FolderNode("InternalAssets", 0));
        }

        #region Folder Tree

        /// <summary>
        /// Represents a folder in the tree.
        /// </summary>
        public class FolderNode
        {
            public FolderNode Parent;
            public DirectoryInfo Info;
            public int Depth;
            public int ChildCount;
            public string VirtualName; // non-null for virtual (non-filesystem) folders

            public bool IsVirtual => VirtualName != null;
            public string DisplayName => VirtualName ?? Info?.Name ?? "";

            public FolderNode(FolderNode parent, DirectoryInfo info, int depth)
            {
                Parent = parent;
                Info = info;
                Depth = depth;
                ChildCount = info.GetDirectories().Length;
            }

            /// <summary>Virtual folder constructor (no filesystem backing).</summary>
            public FolderNode(string virtualName, int depth)
            {
                VirtualName = virtualName;
                Depth = depth;
            }

            public bool Expanded
            {
                get => IsVirtual ? _expandedPaths.Contains(VirtualName) : _expandedPaths.Contains(Info.FullName);
                set
                {
                    var key = IsVirtual ? VirtualName : Info.FullName;
                    if (value) _expandedPaths.Add(key);
                    else _expandedPaths.Remove(key);
                }
            }
        }

        private Control CreateFolderItem(int index)
        {
            var node = _folders[index];

            var row = new Button
            {
                Size = new Point(100, 20),
                Dock = DockStyle.Top,
                Style = "",
                Tag = node
            };

            var indent = new Control
            {
                Style = "node",
                NoEvents = true,
                Dock = DockStyle.Left,
                Size = new Point(12 * node.Depth, 20)
            };

            var foldout = new ImageControl
            {
                Style = "node",
                Size = new Point(20, 20),
                Dock = DockStyle.Left,
                Tiling = TextureMode.Center,
                Color = ColorInt.ARGB(1f, .5f, .5f, .5f)
            };
            foldout.MouseClick += (s, e) =>
            {
                if (e.Button > 0) return;
                var n = row.Tag as FolderNode;
                var idx = _folders.IndexOf(n);
                ToggleFolderExpand(idx, n);
            };

            var icon = new ImageControl
            {
                Style = "node",
                NoEvents = true,
                Size = new Point(18, 20),
                Dock = DockStyle.Left,
                Tiling = TextureMode.Center,
                Texture = "folder.dds",
                Color = ColorInt.ARGB(1f, .5f, .5f, .5f)
            };

            var label = new Label
            {
                Style = "node",
                NoEvents = true,
                Size = new Point(20, 20),
                Dock = DockStyle.Fill,
            };

            row.GetElements().Add(indent);
            row.GetElements().Add(foldout);
            row.GetElements().Add(icon);
            row.GetElements().Add(label);

            row.MouseClick += (s, e) =>
            {
                if (e.Button > 0) return;
                SelectFolder(s.Tag as FolderNode);
            };

            row.AllowFocus = true;

            BindFolderItem(row, index);
            return row;
        }

        private void BindFolderItem(Control control, int index)
        {
            var node = _folders[index];
            control.Tag = node;

            var elems = control.GetElements();
            // indent
            elems[0].Size = new Point(12 * node.Depth, 20);
            // foldout
            var foldout = elems[1] as ImageControl;
            foldout.Enabled = node.ChildCount > 0;
            foldout.NoEvents = node.ChildCount == 0;
            foldout.Texture = node.ChildCount > 0
                ? (node.Expanded ? "nav_down.dds" : "nav_right.dds")
                : "";
            // label
            (elems[3] as Label).Text = node.DisplayName;
            // Selection highlight
            bool isSelected;
            if (node.IsVirtual)
                isSelected = _selectedFolder?.VirtualName == node.VirtualName;
            else
                isSelected = _selectedFolder?.Info?.FullName == node.Info?.FullName;
            (control as Button).State = isSelected ? ControlState.Selected : ControlState.Default;
        }

        private void ToggleFolderExpand(int index, FolderNode node)
        {
            node.Expanded = !node.Expanded;

            if (node.Expanded)
            {
                var children = new List<FolderNode>();
                GatherExpandedChildren(node, children);
                _folders.InsertRange(index + 1, children);
            }
            else
            {
                var children = new List<FolderNode>();
                GatherExpandedChildren(node, children);
                _folders.RemoveRange(index + 1, children.Count);
            }

            _folderList.Refresh();
            _folderList.UpdateVirtualList();
        }

        private void GatherExpandedChildren(FolderNode parent, List<FolderNode> result)
        {
            var dirs = parent.Info.GetDirectories();
            foreach (var dir in dirs)
            {
                var child = new FolderNode(parent, dir, parent.Depth + 1);
                result.Add(child);
                if (child.Expanded)
                    GatherExpandedChildren(child, result);
            }
        }

        private void SelectFolder(FolderNode node)
        {
            _selectedFolder = node;
            _folderList.Refresh();
            CancelRename();

            _dataSource.Clear();

            if (node.IsVirtual)
            {
                BuildInternalAssetsData();
                _breadcrumb.Controls.Clear();
                var lbl = new Label { Text = "InternalAssets", Dock = DockStyle.Left, Style = "node" };
                _breadcrumb.Controls.Add(lbl);
            }
            else
            {
                BuildContentData(node.Info);
                BuildBreadcrumb(node.Info);
            }

            ApplyDataSource();
        }

        /// <summary>
        /// Populate _dataSource with all InternalAssets entries.
        /// </summary>
        private void BuildInternalAssetsData()
        {
            void Add(string name, string typeName, string guid, Type assetType) =>
                _dataSource.Add(new CardData { Name = name, TypeName = typeName, SourceGuid = guid, DragAssetType = assetType, IsSourceCard = true });

            // Effects
            Add("DefaultEffect",   "Effect", IA.Guids.DefaultEffect,   typeof(Effect));
            Add("TerrainEffect",   "Effect", IA.Guids.TerrainEffect,   typeof(Effect));
            Add("DecoratorEffect", "Effect", IA.Guids.DecoratorEffect, typeof(Effect));
            Add("FoliageEffect",   "Effect", IA.Guids.FoliageEffect,   typeof(Effect));
            Add("TrunkEffect",     "Effect", IA.Guids.TrunkEffect,     typeof(Effect));
            Add("SkyboxEffect",    "Effect", IA.Guids.SkyboxEffect,    typeof(Effect));
            Add("TransparentEffect", "Effect", IA.Guids.TransparentEffect, typeof(Effect));
            Add("SkinnedEffect", "Effect", IA.Guids.SkinnedEffect, typeof(Effect));
            Add("CrossmeshEffect", "Effect", IA.Guids.CrossmeshEffect, typeof(Effect));

            // Materials
            Add("DefaultMaterial",   "Material", IA.Guids.DefaultMaterial,   typeof(Material));
            Add("TerrainMaterial",   "Material", IA.Guids.TerrainMaterial,   typeof(Material));
            Add("DecoratorMaterial", "Material", IA.Guids.DecoratorMaterial, typeof(Material));
            Add("FoliageMaterial",   "Material", IA.Guids.FoliageMaterial,   typeof(Material));
            Add("TrunkMaterial",     "Material", IA.Guids.TrunkMaterial,     typeof(Material));
            Add("SkyboxMaterial",    "Material", IA.Guids.SkyboxMaterial,    typeof(Material));
            Add("CrossmeshMaterial", "Material", IA.Guids.CrossmeshMaterial, typeof(Material));

            // Textures
            Add("White",           "Texture", IA.Guids.White,           typeof(Texture));
            Add("Black",           "Texture", IA.Guids.Black,           typeof(Texture));
            Add("FlatNormal",      "Texture", IA.Guids.FlatNormal,      typeof(Texture));
            Add("DefaultDiffuse",  "Texture", IA.Guids.DefaultDiffuse,  typeof(Texture));
            Add("DefaultNormal",   "Texture", IA.Guids.DefaultNormal,   typeof(Texture));
            Add("DefaultSpecular", "Texture", IA.Guids.DefaultSpecular, typeof(Texture));

            // Meshes
            Add("SphereMesh", "Mesh", IA.Guids.SphereMesh, typeof(Mesh));
            Add("CubeMesh", "Mesh", IA.Guids.CubeMesh, typeof(Mesh));

        }

        #endregion

        #region Content Panel

        /// <summary>
        /// Build _dataSource from a filesystem directory (folders + importable files).
        /// </summary>
        private void BuildContentData(DirectoryInfo dir)
        {
            if (dir == null) return;

            // Subdirectories first
            foreach (var sub in dir.GetDirectories())
            {
                _dataSource.Add(new CardData
                {
                    Name = sub.Name,
                    TypeName = "FOLDER",
                    IsFolder = true,
                    FolderInfo = sub,
                    IsSourceCard = true
                });
            }

            // Importable files
            foreach (var file in dir.GetFiles())
            {
                var ext = file.Extension;
                if (string.IsNullOrEmpty(ext)) continue;
                if (ext.Equals(".meta", StringComparison.OrdinalIgnoreCase)) continue;
                if (!AssetDatabase.IsImportableExtension(ext)) continue;

                var fileName = Path.GetFileNameWithoutExtension(file.Name);
                var relativePath = GetRelativePath(file.FullName);
                var guid = AssetDatabase.PathToGuid(relativePath);
                var meta = guid != null ? AssetDatabase.GetMeta(guid) : null;

                var displayType = meta?.MainSemanticType?.ToUpperInvariant()
                    ?? CleanImporterType(meta?.ImporterType)
                    ?? ext.TrimStart('.').ToUpperInvariant();
                var importer = guid != null ? AssetDatabase.GetImporter(guid) : null;
                var importerAssetType = importer?.AssetType;

                // Use semantic type for drag-drop when available (e.g. PCGGraph instead of AssetDefinitionData)
                var dragType = importerAssetType;
                if (!string.IsNullOrEmpty(meta?.MainSemanticType))
                    dragType = AssetDragData.ResolveType(meta.MainSemanticType) ?? dragType;

                _dataSource.Add(new CardData
                {
                    Name = fileName,
                    TypeName = displayType,
                    SourceGuid = guid,
                    DragAssetType = dragType,
                    IsSourceCard = true
                });

                if (meta?.SubAssets?.Count > 0)
                {
                    foreach (var sub in meta.SubAssets)
                    {
                        if (sub.Hidden) continue;
                        var resolvedType = sub.AssetType ?? sub.Type;
                        _dataSource.Add(new CardData
                        {
                            Name = sub.Name,
                            TypeName = CleanTypeName(resolvedType),
                            SourceGuid = guid,
                            SubGuid = sub.Guid,
                            SubType = resolvedType,
                            DragAssetType = AssetDragData.ResolveType(resolvedType),
                            IsSourceCard = false
                        });
                    }
                }
            }

            // Script files (.cs) — not imported, just shown for browsing
            foreach (var file in dir.GetFiles("*.cs"))
            {
                _dataSource.Add(new CardData
                {
                    Name = Path.GetFileNameWithoutExtension(file.Name),
                    TypeName = "SCRIPT",
                    IsSourceCard = true,
                    FullPath = file.FullName
                });
            }
        }

        // GUID to start rename on after next refresh (set by Create)
        private string _pendingRenameGuid;

        /// <summary>
        /// Push the current _dataSource into the VirtualList as rows.
        /// Uses a simple ArrayList of row indices as the VirtualList data source.
        /// </summary>
        private void ApplyDataSource()
        {
            UpdateColumns();
        }

        /// <summary>
        /// Recalculate column count from current panel width and card size, then refresh the VirtualList.
        /// </summary>
        private void UpdateColumns()
        {
            var width = _contentList.ClipFrame.Size.x;
            if (width <= 0) width = 400; // fallback before first layout

            _columns = Math.Max(1, (width + CardSpacing) / (_cardWidth + CardSpacing));

            _contentList.ItemHeight = CardHeight + CardSpacing;

            // Build a row-index list for VirtualList (it needs an IList)
            var rowList = new System.Collections.ArrayList();
            var rowCount = RowCount;
            for (int i = 0; i < rowCount; i++)
                rowList.Add(i);

            _contentList.DataSource = rowList;
            _contentList.Refresh();
        }

        /// <summary>
        /// VirtualList CreateItem callback — builds a reusable row Frame.
        /// </summary>
        private Control CreateRow(int index)
        {
            var row = new Frame
            {
                Size = new Point(100, CardHeight + CardSpacing),
                Dock = DockStyle.Top,
                Style = "",
            };

            // Pre-populate with _columns cards
            for (int c = 0; c < _columns; c++)
            {
                var card = CreateCard();
                row.Controls.Add(card);
            }

            BindRow(row, index);
            return row;
        }

        /// <summary>
        /// VirtualList BindItem callback — rebinds each card in the row.
        /// </summary>
        private void BindRow(Control control, int index)
        {
            var row = control as Frame;
            if (row == null) return;

            int startIndex = index * _columns;

            // Ensure the row has exactly _columns card controls
            while (row.Controls.Count < _columns)
                row.Controls.Add(CreateCard());

            while (row.Controls.Count > _columns)
            {
                var extra = row.Controls[row.Controls.Count - 1];
                row.Controls.Remove(extra);
            }

            for (int c = 0; c < _columns; c++)
            {
                var card = row.Controls[c] as Button;
                if (card == null) continue;

                int dataIndex = startIndex + c;
                if (dataIndex < _dataSource.Count)
                {
                    card.Visible = true;
                    BindCard(card, dataIndex);
                }
                else
                {
                    card.Visible = false;
                }
            }
        }

        /// <summary>
        /// Create a single reusable card Button with element slots:
        /// [0] = icon (ImageControl), [1] = nameLabel, [2] = stripe, [3] = typeLabel
        /// </summary>
        private Button CreateCard()
        {
            var card = new Button
            {
                Size = new Point(_cardWidth, CardHeight),
                Style = "tile",
                Dock = DockStyle.Left,
                Margin = new Margin(0, 0, CardSpacing, CardSpacing),
            };

            // Click handler — reads data index from Tag
            card.MouseClick += OnCardClick;
            card.KeyDown += OnCardKeyDown;
            card.MouseDrag += OnCardDrag;
            card.MouseDoubleClick += OnCardDoubleClick;
            card.AllowFocus = true;

            var icon = new ImageControl
            {
                Size = new Point(_cardWidth, _cardWidth),
                Dock = DockStyle.Top,
                NoEvents = true,
                Style = "dark2",
            };
            card.GetElements().Add(icon);

            var nameLabel = new Label
            {
                Size = new Point(100, 16),
                Dock = DockStyle.Top,
                Margin = new Margin(2, 4, 2, 0),
                NoEvents = true
            };
            card.GetElements().Add(nameLabel);

            var stripe = new Frame
            {
                Size = new Point(0, 4),
                Dock = DockStyle.Bottom,
                NoEvents = true,
            };
            card.GetElements().Add(stripe);

            var typeLabel = new Label
            {
                TextAlign = Alignment.MiddleRight,
                Size = new Point(100, 14),
                Dock = DockStyle.Bottom,
                UseTextColor = true,
                Margin = new Margin(2, 0, 4, 2),
                TextColor = ColorInt.ARGB(1, .5f, .5f, .5f),
                NoEvents = true
            };
            card.GetElements().Add(typeLabel);

            return card;
        }

        /// <summary>
        /// Rebind a card control to data at the given index.
        /// </summary>
        private void BindCard(Button card, int dataIndex)
        {
            var data = _dataSource[dataIndex];
            card.Size = new Point(_cardWidth, CardHeight);
            card.Tooltip = data.Name;

            // Store the data index for event handlers
            card.Tag = dataIndex;

            var elems = card.GetElements();
            // [0] icon
            var icon = elems[0] as ImageControl;
            icon.Size = new Point(_cardWidth, _cardWidth);

            if (data.IsFolder)
            {
                icon.Texture = "icon_folder.png";
                icon.Tiling = TextureMode.Center;
                icon.Color = ColorInt.ARGB(1f, .6f, .6f, .6f);
            }
            else
            {
                var thumbGuid = data.SubGuid ?? data.SourceGuid;
                var thumbTexture = AssetDatabase.GetThumbnail(thumbGuid);
                var hasThumbnail = !string.IsNullOrEmpty(thumbTexture) && thumbTexture != (InternalAssets.Gray?.Name ?? "");

                icon.Texture = thumbTexture ?? "";
                icon.Tiling = hasThumbnail ? TextureMode.Stretch : TextureMode.Center;
                icon.Color = hasThumbnail ? ColorInt.ARGB(1f, 1f, 1f, 1f) : ColorInt.ARGB(1f, .4f, .4f, .4f);
            }

            // [1] name label
            (elems[1] as Label).Text = data.Name;

            // [2] stripe
            elems[2].Style = data.IsFolder ? "" : EditorPreferences.Instance.GetAssetStyleName(data.TypeName);

            // [3] type label
            (elems[3] as Label).Text = data.TypeName;

            // Handle pending rename
            if (data.IsSourceCard && !string.IsNullOrEmpty(_pendingRenameGuid) && data.SourceGuid == _pendingRenameGuid)
            {
                _pendingRenameGuid = null;
                VoidEvent autoRename = null;
                autoRename = (s) =>
                {
                    card.Update -= autoRename;
                    if (_renamingCard != null) return;
                    BeginRename(card, data.SourceGuid);
                };
                card.Update += autoRename;
            }
        }

        // ── Card event handlers (read data index from Tag) ──

        private void OnCardClick(Control sender, MouseEventArgs args)
        {
            if (sender.Tag is not int dataIndex || dataIndex >= _dataSource.Count) return;
            var data = _dataSource[dataIndex];

            if (args.Button > 0)
            {
                if (data.IsSourceCard && !data.IsFolder && !string.IsNullOrEmpty(data.SourceGuid))
                {
                    _selectedCard = sender as Button;
                    _selectedCardGuid = data.SourceGuid;
                    ShowAssetContextMenu(data.SourceGuid, sender as Button);
                }
                return;
            }

            _selectedCard = sender as Button;
            _selectedCardGuid = data.IsSourceCard ? data.SourceGuid : null;

            if (!data.IsFolder)
                SelectAsset(data.SourceGuid, data.SubGuid);
        }

        private void OnCardDoubleClick(Control sender, MouseEventArgs args)
        {
            if (args.Button > 0) return;
            if (sender.Tag is not int dataIndex || dataIndex >= _dataSource.Count) return;
            var data = _dataSource[dataIndex];

            if (data.IsFolder && data.FolderInfo != null)
            {
                ExpandToFolder(data.FolderInfo);
                return;
            }

            // Script files: open in Visual Studio with the script project
            if (!string.IsNullOrEmpty(data.FullPath))
            {
                ScriptCompiler.OpenInVisualStudio(data.FullPath);
                return;
            }

            // Try to open in a custom editor
            var guid = data.SubGuid ?? data.SourceGuid;
            var assetType = data.DragAssetType;

            // For source cards from .asset files, resolve type from meta
            if (assetType == null && !string.IsNullOrEmpty(data.SourceGuid))
            {
                var meta = AssetDatabase.GetMeta(data.SourceGuid);
                if (meta != null)
                {
                    // Simple asset: use semantic type (e.g. "PCGGraph")
                    if (!string.IsNullOrEmpty(meta.MainSemanticType))
                        assetType = AssetDragData.ResolveType(meta.MainSemanticType);
                    // Compound asset: use first sub-asset type
                    else if (meta.SubAssets?.Count > 0)
                        assetType = AssetDragData.ResolveType(meta.SubAssets[0].AssetType ?? meta.SubAssets[0].Type);
                }
            }

            if (assetType == null || string.IsNullOrEmpty(guid)) return;

            MessageDispatcher.Send(Msg.OpenAssetInEditor, (guid, assetType));
        }

        private void OnCardKeyDown(Control sender, KeyEventArgs args)
        {
            if (sender.Tag is not int dataIndex || dataIndex >= _dataSource.Count) return;
            var data = _dataSource[dataIndex];

            if (args.Key == Squid.Keys.F2 && data.IsSourceCard && !data.IsFolder && !string.IsNullOrEmpty(data.SourceGuid))
                BeginRename(sender as Button, data.SourceGuid);
        }

        private void OnCardDrag(Control sender, MouseEventArgs args)
        {
            if (sender.Tag is not int dataIndex || dataIndex >= _dataSource.Count) return;
            var data = _dataSource[dataIndex];
            if (data.IsFolder) return;

            var dragGuid = data.SubGuid ?? data.SourceGuid;
            var dragType = data.DragAssetType ?? AssetDragData.ResolveType(data.SubType);
            var dragData = new AssetDragData { Guid = dragGuid, AssetType = dragType };

            var proxy = new Label
            {
                Text = data.Name,
                Size = new Point(_cardWidth, _cardWidth),
                Style = "tooltip",
                Tag = dragData,
                NoEvents = true
            };

            proxy.Position = Gui.MousePosition - proxy.Size / 2;
            DoDragDrop(proxy);
        }

        /// <summary>
        /// Resolve what should be inspected for a given asset selection
        /// and dispatch it to the InspectorControl.
        /// </summary>
        private void SelectAsset(string sourceGuid, string subGuid)
        {
            if (string.IsNullOrEmpty(sourceGuid)) return;

            object target = null;

            if (!string.IsNullOrEmpty(subGuid))
            {
                // Subasset: load the cached artifact by GUID
                var meta = AssetDatabase.GetMeta(sourceGuid);
                var sub = meta?.SubAssets?.Find(s => s.Guid == subGuid);
                if (sub != null)
                    target = LoadSubAsset(sub);
            }
            else
            {
                // Source asset: let the importer decide what to inspect
                var meta = AssetDatabase.GetMeta(sourceGuid);
                var importer = AssetDatabase.GetImporter(sourceGuid);
                if (importer != null && meta != null)
                    target = importer.GetInspectionTarget(meta);
            }

            if (target != null)
                Selector.SelectedObject = target;
        }

        /// <summary>
        /// Load a subasset by its type for inspection.
        /// </summary>
        private static object LoadSubAsset(SubAssetEntry sub)
        {
            try
            {
                var assetType = AssetDragData.ResolveType(sub.AssetType ?? sub.Type);
                if (assetType != null && assetType != typeof(Asset))
                    return Engine.Assets.LoadByGuid(sub.Guid, assetType);
            }
            catch (Exception ex)
            {
                Debug.Log($"[AssetBrowser] Failed to load subasset '{sub.Name}' ({sub.Type}): {ex.Message}");
            }

            return null;
        }

        // ── Context Menu ──

        private void ShowAssetContextMenu(string sourceGuid, Button card)
        {
            var desktop = Desktop;
            if (desktop == null) return;

            var menu = new Window
            {
                Style = "frame",
                AutoSize = AutoSize.Vertical,
                Size = new Point(180, 0),
                MinSize = new Point(180, 0),
                Position = new Point(Gui.MousePosition.x, Gui.MousePosition.y),
            };

            AddContextMenuItem(menu, "Rename", () => { Desktop.CloseDropdowns(); BeginRename(card, sourceGuid); });
            AddContextMenuItem(menu, "Show in Explorer", () =>
            {
                Desktop.CloseDropdowns();

                var path = AssetDatabase.GuidToPath(sourceGuid);
                if (!string.IsNullOrEmpty(path))
                {
                    var fullPath = Path.Combine(AssetDatabase.Project.AssetsDirectory, path);
                    if (File.Exists(fullPath))
                        System.Diagnostics.Process.Start("explorer.exe", $"/select,\"{fullPath}\"");
                }
            });
            AddContextMenuItem(menu, "Reimport", () =>
            {
                Desktop.CloseDropdowns();
                var path = AssetDatabase.GuidToPath(sourceGuid);
                if (!string.IsNullOrEmpty(path))
                {
                    AssetDatabase.ImportAssetByPath(path);
                    RefreshCurrentFolder();
                    Debug.Log($"[AssetBrowser] Reimported: {path}");
                }
            });

            menu.PerformLayout();
            desktop.ShowDropdown(menu, false);
        }

        private void AddContextMenuItem(Window menu, string text, Action action)
        {
            var item = new Button
            {
                Style = "button",
                Size = new Point(180, 28),
                Dock = DockStyle.Top,
                Text = text,
            };
            item.MouseClick += (s, e) =>
            {
                if (e.Button > 0) return;
                action();
            };
            menu.Controls.Add(item);
        }

        // ── Inline Rename ──

        private void BeginRename(Button card, string sourceGuid)
        {
            if (card == null || string.IsNullOrEmpty(sourceGuid)) return;

            // Cancel any existing rename
            CancelRename();

            _renamingCard = card;
            _renamingGuid = sourceGuid;

            // Find the name label (element index 1 = nameLabel)
            var elems = card.GetElements();
            if (elems.Count < 2) return;
            var nameLabel = elems[1] as Label;
            if (nameLabel == null) return;

            var currentName = nameLabel.Text;

            // Create a TextBox overlaying the name label
            _renameBox = new TextBox
            {
                Text = currentName,
                Style = "textbox",
                Size = nameLabel.Size,
                Dock = DockStyle.Top,
                Margin = nameLabel.Margin,
            };

            // Hide original label, show textbox
            nameLabel.Visible = false;
            elems.Insert(elems.IndexOf(nameLabel) + 1, _renameBox);

            _renameBox.Focus();
            _renameBox.SelectAll();

            // Commit on text commit (click elsewhere)
            _renameBox.TextCommit += (s, e) => CommitRename();
            _renameBox.TextCancel += (s, e) => CancelRename();
        }

        private void CommitRename()
        {
            if (_renameBox == null || _renamingCard == null) return;

            var newName = _renameBox.Text?.Trim();
            var guid = _renamingGuid;

            // Validate
            bool valid = !string.IsNullOrEmpty(newName);
            if (valid)
            {
                // Check for invalid filename characters
                foreach (var c in Path.GetInvalidFileNameChars())
                {
                    if (newName.Contains(c))
                    {
                        valid = false;
                        break;
                    }
                }
            }

            // Clean up UI first (avoid re-entrant calls from LostFocus)
            var card = _renamingCard;
            var box = _renameBox;
            _renameBox = null;
            _renamingCard = null;
            _renamingGuid = null;

            // Restore label visibility
            var elems = card.GetElements();
            elems.Remove(box);
            foreach (var elem in elems)
            {
                if (elem is Label lbl && lbl.Dock == DockStyle.Top)
                {
                    lbl.Visible = true;
                    if (valid) lbl.Text = newName;
                    break;
                }
            }

            // Perform rename
            if (valid && !string.IsNullOrEmpty(guid))
            {
                AssetDatabase.RenameAsset(guid, newName);
                // Refresh the browser to reflect the new name everywhere
                RefreshCurrentFolder();
            }
        }

        private void CancelRename()
        {
            if (_renameBox == null || _renamingCard == null) return;

            var card = _renamingCard;
            var box = _renameBox;
            _renameBox = null;
            _renamingCard = null;
            _renamingGuid = null;

            var elems = card.GetElements();
            elems.Remove(box);
            foreach (var elem in elems)
            {
                if (elem is Label lbl && lbl.Dock == DockStyle.Top)
                {
                    lbl.Visible = true;
                    break;
                }
            }
        }

        public void RefreshCurrentFolder()
        {
            _dataSource.Clear();
            if (_selectedFolder != null)
            {
                if (_selectedFolder.IsVirtual)
                    BuildInternalAssetsData();
                else
                    BuildContentData(_selectedFolder.Info);
            }
            ApplyDataSource();
        }

        #endregion

        #region Breadcrumb

        private void BuildBreadcrumb(DirectoryInfo dir)
        {
            _breadcrumb.Controls.Clear();

            var assetsDir = AssetDatabase.Project?.AssetsDirectory;
            if (assetsDir == null) return;

            // Build path segments from Assets/ root to current dir
            var segments = new List<DirectoryInfo>();
            var current = dir;
            while (current != null)
            {
                segments.Add(current);
                if (current.FullName.Equals(assetsDir, StringComparison.OrdinalIgnoreCase))
                    break;
                current = current.Parent;
            }
            segments.Reverse();

            for (int i = 0; i < segments.Count; i++)
            {
                if (i > 0)
                {
                    var sep = new Label
                    {
                        Text = ">",
                        NoEvents = true,
                        Size = new Point(16, 26),
                        Dock = DockStyle.Left,
                        TextAlign = Alignment.MiddleCenter,
                        UseTextColor = true,
                        TextColor = ColorInt.ARGB(1, .4f, .4f, .4f),
                    };
                    _breadcrumb.Controls.Add(sep);
                }

                var seg = segments[i];
                var btn = new Button
                {
                    Text = seg.Name,
                    AutoSize = AutoSize.Horizontal,
                    Dock = DockStyle.Left,
                    Tag = seg,
                    Style = "node"
                };
                btn.MouseClick += (s, e) =>
                {
                    var d = s.Tag as DirectoryInfo;
                    ExpandToFolder(d);
                };
                _breadcrumb.Controls.Add(btn);
            }
        }

        #endregion

        #region Search

        private void OnSearchChanged(Control sender)
        {
            var query = _searchBox.Text;
            _dataSource.Clear();

            if (string.IsNullOrWhiteSpace(query))
            {
                // Restore current folder view
                if (_selectedFolder != null)
                {
                    if (_selectedFolder.IsVirtual)
                        BuildInternalAssetsData();
                    else
                        BuildContentData(_selectedFolder.Info);
                    BuildBreadcrumb(_selectedFolder.Info);
                }
                ApplyDataSource();
                return;
            }

            // Search all files under Assets/ matching the query
            var assetsDir = AssetDatabase.Project?.AssetsDirectory;
            if (assetsDir == null || !Directory.Exists(assetsDir)) return;

            var files = Directory.GetFiles(assetsDir, $"*{query}*", SearchOption.AllDirectories);
            foreach (var filePath in files)
            {
                var ext = Path.GetExtension(filePath);
                if (string.IsNullOrEmpty(ext)) continue;
                if (ext.Equals(".meta", StringComparison.OrdinalIgnoreCase)) continue;
                if (!AssetDatabase.IsImportableExtension(ext)) continue;

                var fileName = Path.GetFileNameWithoutExtension(filePath);
                var relativePath = GetRelativePath(filePath);
                var guid = AssetDatabase.PathToGuid(relativePath);
                var meta = guid != null ? AssetDatabase.GetMeta(guid) : null;

                var displayType = meta?.MainSemanticType?.ToUpperInvariant()
                    ?? CleanImporterType(meta?.ImporterType)
                    ?? ext.TrimStart('.').ToUpperInvariant();
                var importer = guid != null ? AssetDatabase.GetImporter(guid) : null;
                var importerAssetType = importer?.AssetType;

                // Use semantic type for drag-drop when available
                var dragType = importerAssetType;
                if (!string.IsNullOrEmpty(meta?.MainSemanticType))
                    dragType = AssetDragData.ResolveType(meta.MainSemanticType) ?? dragType;

                _dataSource.Add(new CardData
                {
                    Name = fileName,
                    TypeName = displayType,
                    SourceGuid = guid,
                    DragAssetType = dragType,
                    IsSourceCard = true
                });

                if (meta?.SubAssets?.Count > 0)
                {
                    foreach (var sub in meta.SubAssets)
                    {
                        if (sub.Hidden) continue;
                        var resolvedType = sub.AssetType ?? sub.Type;
                        _dataSource.Add(new CardData
                        {
                            Name = sub.Name,
                            TypeName = CleanTypeName(resolvedType),
                            SourceGuid = guid,
                            SubGuid = sub.Guid,
                            SubType = resolvedType,
                            DragAssetType = AssetDragData.ResolveType(resolvedType),
                            IsSourceCard = false
                        });
                    }
                }
            }

            ApplyDataSource();
        }

        #endregion

        #region Helpers

        private void ExpandToFolder(DirectoryInfo directory)
        {
            var target = _folders.Find(f => f.Info?.FullName == directory.FullName);
            if (target != null)
            {
                SelectFolder(target);
                _folderList.ScrollTo(_folders.IndexOf(target));
                return;
            }

            // Need to expand parents first
            var assetsDir = AssetDatabase.Project?.AssetsDirectory;
            if (assetsDir == null) return;

            // Set all ancestors as expanded
            var current = directory;
            while (current != null && !current.FullName.Equals(assetsDir, StringComparison.OrdinalIgnoreCase))
            {
                _expandedPaths.Add(current.FullName);
                current = current.Parent;
            }

            // Rebuild the tree (includes InternalAssets virtual node)
            RebuildFolderTree(assetsDir);

            _folderList.Refresh();
            _folderList.UpdateVirtualList();

            target = _folders.Find(f => f.Info.FullName == directory.FullName);
            if (target != null)
            {
                SelectFolder(target);
                _folderList.ScrollTo(_folders.IndexOf(target));
            }
        }

        private string GetRelativePath(string fullPath)
        {
            var assetsDir = AssetDatabase.Project?.AssetsDirectory;
            if (assetsDir == null) return fullPath;

            var root = assetsDir.EndsWith(Path.DirectorySeparatorChar.ToString())
                ? assetsDir : assetsDir + Path.DirectorySeparatorChar;

            if (fullPath.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                return fullPath[root.Length..].Replace('\\', '/');

            return fullPath;
        }

        /// <summary>
        /// Clean an artifact type name for display: strip "Data" suffix, uppercase.
        /// </summary>
        private static string CleanTypeName(string type)
        {
            if (string.IsNullOrEmpty(type)) return null;
            if (type.EndsWith("Data", StringComparison.Ordinal))
                type = type[..^4];
            return type.ToUpperInvariant();
        }

        /// <summary>
        /// Derive a clean type name from a fully-qualified importer type.
        /// e.g. "Freefall.Assets.Importers.ModelImporter" → "MODEL"
        /// </summary>
        private static string CleanImporterType(string importerType)
        {
            if (string.IsNullOrEmpty(importerType)) return null;
            var name = importerType;
            var dot = name.LastIndexOf('.');
            if (dot >= 0) name = name[(dot + 1)..];
            if (name.EndsWith("Importer", StringComparison.Ordinal))
                name = name[..^8];
            return name.ToUpperInvariant();
        }

        #endregion

        // ── Resize detection ──
        private int _lastContentWidth;

        protected override void OnUpdate()
        {
            var w = _contentList.ClipFrame.Size.x;
            if (w != _lastContentWidth && w > 0)
            {
                _lastContentWidth = w;
                UpdateColumns();
            }
        }
    }
}
