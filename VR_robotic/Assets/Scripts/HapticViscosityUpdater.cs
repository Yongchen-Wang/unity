using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class HapticViscosityUpdater : MonoBehaviour
{
    [Header("Moving Objects (used to calculate distance)")]
    public GameObject movable1;
    public GameObject movable2;

    [Header("Fixed Reference Object")]
    public GameObject referenceObject;

    [Header("Objects With HapticMaterial")]
    public GameObject materialObject1;
    public GameObject materialObject2;

    private HapticMaterial material1;
    private HapticMaterial material2;
    private Collider referenceCollider;

    void Start()
    {
        if (materialObject1 != null)
            material1 = materialObject1.GetComponent<HapticMaterial>();

        if (materialObject2 != null)
            material2 = materialObject2.GetComponent<HapticMaterial>();

        if (material1 == null || material2 == null)
        {
            Debug.LogError("Missing HapticMaterial on one of the material objects.");
        }

        if (movable1 == null || movable2 == null || referenceObject == null)
        {
            Debug.LogError("One or more required GameObjects are not assigned.");
        }

        referenceCollider = referenceObject.GetComponent<Collider>();
        if (referenceCollider == null)
        {
            Debug.LogError("Reference object is missing a Collider component.");
        }
    }

    void Update()
    {
        if (material1 != null && movable1 != null)
        {
            float distance1 = GetDistanceToSurface(movable1);
            material1.hViscosity = ComputeViscosity(distance1);
        }

        if (material2 != null && movable2 != null)
        {
            float distance2 = GetDistanceToSurface(movable2);
            material2.hViscosity = ComputeViscosity(distance2);
        }
    }

    float GetDistanceToSurface(GameObject movingObject)
    {
        if (referenceCollider == null) return float.MaxValue;

        Vector3 closestPoint = referenceCollider.ClosestPoint(movingObject.transform.position);
        return Vector3.Distance(movingObject.transform.position, closestPoint);
    }

    /// <summary>
    /// Maps distance [0.4, 1.0] to viscosity [1, 0]
    /// </summary>
    float ComputeViscosity(float distance)
    {
        float minRange = 0.4f;
        float maxRange = 1.0f;

        if (distance <= minRange)
            return 1f;
        else if (distance >= maxRange)
            return 0f;
        else
            return 1f - (distance - minRange) / (maxRange - minRange);  // œﬂ–‘≤Â÷µ
    }
}
