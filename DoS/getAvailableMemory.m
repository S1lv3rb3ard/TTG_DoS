function memAvailBytes = getAvailableMemory()
    % Returns available system memory in bytes across Windows, Linux, and macOS
    
    if ispc
        % Windows: Use the built-in memory function
        [~, sys] = memory;
        memAvailBytes = sys.PhysicalMemory.Available;
        
    elseif ismac
        % macOS: Query vm_stat for free pages (page size is typically 4096 bytes)
        [status, cmdOut] = system('vm_stat | grep "Pages free"');
        if status == 0
            % Extract the raw number using regular expressions
            pagesFree = str2double(regexp(cmdOut, '\d+', 'match', 'once'));
            memAvailBytes = pagesFree * 4096; 
        else
            warning('Could not determine Mac memory. Defaulting to 2GB safety limit.');
            memAvailBytes = 2 * 1024^3;
        end
        
    elseif isunix
        % Linux: Read /proc/meminfo to get 'MemAvailable' (returns kilobytes)
        [status, cmdOut] = system('awk ''/MemAvailable/ {print $2}'' /proc/meminfo');
        
        if status == 0 && ~isempty(strtrim(cmdOut))
            memAvailBytes = str2double(cmdOut) * 1024;
        else
            % Fallback for older Linux kernels that only output 'MemFree'
            [statFree, outFree] = system('awk ''/MemFree/ {print $2}'' /proc/meminfo');
            if statFree == 0 && ~isempty(strtrim(outFree))
                memAvailBytes = str2double(outFree) * 1024;
            else
                warning('Could not determine Linux memory. Defaulting to 2GB safety limit.');
                memAvailBytes = 2 * 1024^3;
            end
        end
        
    else
        warning('Unknown OS. Defaulting to 2GB safety limit.');
        memAvailBytes = 2 * 1024^3;
    end
end