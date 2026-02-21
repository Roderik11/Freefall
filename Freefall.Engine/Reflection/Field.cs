using System;
using System.ComponentModel;
using System.Reflection;

namespace Freefall.Reflection
{
    /// <summary>
    /// Unified wrapper over FieldInfo and PropertyInfo.
    /// Provides a common interface for get/set, type queries, and attribute access.
    /// Used by Reflector, YAMLSerializer, and GUIObject.
    /// </summary>
    public class Field
    {
        public MemberInfo Member;

        private readonly PropertyInfo _prop;
        private readonly FieldInfo _field;

        public bool CanWrite { get; }
        public bool Ignored { get; }
        public bool Browsable { get; }
        public bool IsPublic { get; }

        public string Category = string.Empty;
        public string Description = string.Empty;

        public Field(MemberInfo member)
        {
            Browsable = true;
            Member = member;

            _field = member as FieldInfo;
            _prop = member as PropertyInfo;

            if (_prop != null)
                CanWrite = _prop.GetSetMethod() != null;

            if (_field != null)
                CanWrite = !_field.IsInitOnly;

            if (_field != null && (_field.IsInitOnly || _field.IsLiteral))
                Ignored = true;

            var nonserialized = GetAttribute<NonSerializedAttribute>();
            if (nonserialized != null) Ignored = true;

            var browsable = GetAttribute<BrowsableAttribute>();
            if (browsable != null) Browsable = browsable.Browsable;

            if (_prop != null)
                IsPublic = true;

            if (_field != null)
                IsPublic = _field.IsPublic;

            if (IsInternal(member))
                IsPublic = false;
        }

        public string Name => Member.Name;

        public Type Type
        {
            get
            {
                if (_field != null) return _field.FieldType;
                if (_prop != null) return _prop.PropertyType;
                return null;
            }
        }

        public Type DeclaringType => Member.DeclaringType;

        public T GetAttribute<T>() where T : Attribute
        {
            var att = Member.GetCustomAttributes(typeof(T), false);
            if (att.Length > 0)
                return att[0] as T;
            return null;
        }

        public object[] GetCustomAttributes(Type type, bool inherit)
        {
            if (_field != null) return _field.GetCustomAttributes(type, inherit);
            if (_prop != null) return _prop.GetCustomAttributes(type, inherit);
            return Array.Empty<object>();
        }

        public object GetValue(object instance)
        {
            if (_field != null) return _field.GetValue(instance);
            if (_prop != null) return _prop.GetValue(instance, null);
            return null;
        }

        public void SetValue(object instance, object value)
        {
            if (_field != null) _field.SetValue(instance, value);
            else if (_prop != null) _prop.SetValue(instance, value, null);
        }

        private static bool IsInternal(MemberInfo memberInfo)
        {
            if (memberInfo == null)
                return false;

            return memberInfo switch
            {
                FieldInfo fi => fi.IsAssembly,
                PropertyInfo pi => IsInternal(pi.GetGetMethod(true)) || IsInternal(pi.GetSetMethod(true)),
                MethodInfo mi => mi.IsAssembly,
                _ => false,
            };
        }

        private static bool IsInternal(MethodInfo method)
        {
            return method != null && method.IsAssembly;
        }
    }
}
