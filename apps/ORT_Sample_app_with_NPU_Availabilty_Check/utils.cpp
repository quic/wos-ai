// Copyright (c) Qualcomm Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <sstream> // For std::ostringstream
#include <iostream>
#include <string>
#include <stdint.h>
#include "utils.h"

DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU, 0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed);

struct DriverVersion
{
    union
    {
        struct
        {
            uint16_t d;
            uint16_t c;
            uint16_t b;
            uint16_t a;
        } parts;
        uint64_t value;
    };

    DriverVersion() = default;

    explicit DriverVersion(uint64_t value)
        : value(value)
    {
    }

    DriverVersion(uint16_t a, uint16_t b, uint16_t c, uint16_t d)
    {
        parts.a = a;
        parts.b = b;
        parts.c = c;
        parts.d = d;
    }
};

std::string npuAvailabilityCheck()
{
    std::ostringstream output; // For consolidating all information
    HRESULT hr;
    IDXCoreAdapterFactory* pAdapterFactory = NULL;
    IDXCoreAdapterList* pAdapterList = NULL;
    unsigned int NumAdapters = 0;
    IDXCoreAdapter* pAdapter = NULL;
    char NpuDescription[MAX_PATH];
    LARGE_INTEGER driverVersion;

    // Create the adapter factory
    hr = DXCoreCreateAdapterFactory(__uuidof(IDXCoreAdapterFactory), (void**)&pAdapterFactory);
    if (FAILED(hr))
    {
        return "Failed to create Adapter Factory.\n";
    }

    // Create the adapter list for NPUs
    hr = pAdapterFactory->CreateAdapterList(1, &DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU, __uuidof(IDXCoreAdapterList), (void**)&pAdapterList);
    if (FAILED(hr))
    {
        return "Failed to create Adapter List.\n";
    }

    // Get the number of adapters
    NumAdapters = pAdapterList->GetAdapterCount();
    output << "Number of Adapters: " << NumAdapters << "\n";

    if (NumAdapters == 0)
    {
        output << "Snapdragon NPU not present.\n";
    }
    else
    {
        // Get the first adapter
        hr = pAdapterList->GetAdapter(0, &pAdapter);

        // Get the NPU description
        hr = pAdapter->GetProperty(DXCoreAdapterProperty::DriverDescription, MAX_PATH, NpuDescription);
        if (SUCCEEDED(hr))
        {
            output << "NPU Found: " << NpuDescription << "\n";
        }
        else
        {
            output << "Failed to retrieve NPU description.\n";
        }

        // Get the driver version
        hr = pAdapter->GetProperty(DXCoreAdapterProperty::DriverVersion, sizeof(driverVersion), &driverVersion);
        if (SUCCEEDED(hr))
        {
            DriverVersion ver(driverVersion.QuadPart);
            output << "Driver Version: " << ver.parts.a << "." << ver.parts.b << "." << ver.parts.c << "." << ver.parts.d << "\n";
        }
        else
        {
            output << "Failed to retrieve Driver Version.\n";
        }

        // Query HardwareID
        DXCoreHardwareID hardwareID;
        hr = pAdapter->GetProperty(DXCoreAdapterProperty::HardwareID, sizeof(hardwareID), &hardwareID);
        if (SUCCEEDED(hr))
        {
            output << "Hardware Details:\n";
            output << "  Vendor ID: " << hardwareID.vendorID << "\n";
            output << "  Device ID: " << hardwareID.deviceID << "\n";
            output << "  Subsystem ID: " << hardwareID.subSysID << "\n";
            output << "  Revision: " << hardwareID.revision << "\n";
        }
        else
        {
            output << "Failed to retrieve Hardware Details.\n";
        }

        // Release the adapter
        pAdapter->Release();
    }

    // Release the adapter list and factory
    pAdapterList->Release();
    pAdapterFactory->Release();

    return output.str(); // Return consolidated information as a string
}
