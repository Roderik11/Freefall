using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Security.AccessControl;
using Freefall.Base;

namespace Freefall.Editor
{
    public static class Selector
    {
        private static Entity _selectedEntity;
        private static object _selectedObject;

        private static readonly List<Entity> _selection;
        private static readonly ReadOnlyCollection<Entity> _selectionReadonly;

        public static ReadOnlyCollection<Entity> Selection => _selectionReadonly;
      
        public static object SelectedObject
        {
            get { return _selectedObject; }
            set
            {

                _selectedObject = value;
                _selectedEntity = value as Entity;

                _selection.Clear();
                if (_selectedEntity != null)
                    _selection.Add(_selectedEntity);

                MessageDispatcher.Send(Msg.SelectionChanged, _selectedObject);
            }
        }

        public static Entity SelectedEntity
        {
            get { return _selectedEntity; }
            set { SelectedObject = value; }
        }

        static Selector()
        {
            _selection = new List<Entity>();
            _selectionReadonly = new ReadOnlyCollection<Entity>(_selection);
        }

        public static void SelectOrDeselect(Entity entity)
        {
            if (Selection.Count == 0)
            {
                SelectedEntity = entity;
                return;
            }

            if (Selection.Contains(entity))
                _selection.Remove(entity);
            else
                _selection.Add(entity);

            MessageDispatcher.Send(Msg.SelectionChanged, _selectedObject);
        }

        public static void AddToSelection(Entity entity)
        {
            if (Selection.Contains(entity))
                return;

            if (Selection.Count == 0)
            {
                SelectedEntity = entity;
                return;
            }

            _selection.Add(entity);

            MessageDispatcher.Send(Msg.SelectionChanged, _selectedObject);
        }
    }
}
