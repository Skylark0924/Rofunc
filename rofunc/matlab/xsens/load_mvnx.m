%Convert mvnx file into a MATLAB structure
function [mvnx] = load_mvnx(file)
    %check for existence
    if (exist(file,'file') == 0)
        %Perhaps the mvnx extension was omitted from the file name. Add the
        %extension and try again.
        if (~contains(file,'.mvnx'))
            file = [file '.mvnx'];
        end

        if (exist(file,'file') == 0)
            error(['The file ' file ' could not be found']);
        end
    end

    % show progress bar
    wb = waitbar(0, ['Loading file ', file]);
    onCleanup(@closewb);
    function closewb()
        close(wb);
    end

    %read the mvnx file
    xDoc = xmlread(file);

    %parse xDoc into a MATLAB structure
    mvnx = struct;
    [mvnx] = parseChildNodes(xDoc, mvnx, wb);
end

% ----- Subfunction parseChildNodes -----
function [Data] = parseChildNodes(theNode, Data, wb)
    % Recurse over node children.
    if hasChildNodes(theNode)
        childNodes = getChildNodes(theNode);
        numChildNodes = getLength(childNodes);

        for count = 1:numChildNodes
            theChild = item(childNodes,count-1);
            [name,Data] = getNodeData(theChild, Data, wb);

            if (~strcmp(name,'#text') && ~strcmp(name,'#comment') && ~strcmp(name,'#cdata_dash_section'))
                if (~strcmp(name,'mvnx') && ~strcmp(name,'subject'))
                    text = toCharArray(getTextContent(theChild))';
                    if (~isempty(text) && ~isstruct(text))
                        Data.metaData.(name) = text;
                    end
                end
            end
        end
    end
end

% ----- Subfunction parseSegmentChildNodes -----
function [children,Data] = parseSegmentChildNodes(theNode, Data, wb)
    % Recurse over node children.
    children = struct;
    if hasChildNodes(theNode)
        childNodes = getChildNodes(theNode);
        numChildNodes = getLength(childNodes);

        myName = toCharArray(getNodeName(theNode))';
        for count = 1:numChildNodes
            theChild = item(childNodes,count-1);
            [name,childs,Data] = getSegmentNodeData(theChild, Data, numChildNodes, myName, wb);

            if (~strcmp(name,'#text') && ~strcmp(name,'#comment'))
                if (isempty(fieldnames(childs)))
                    text = toCharArray(getTextContent(theChild))';
                    if(~isempty(text))
                        children.(name) = text;
                    end
                else
                    children.(name) = childs;
                end
            end
        end
    end
end

% ----- Subfunction parseSegmentChildNodesPoints -----
function [children,Data] = parseSegmentChildNodesPoints(theNode, Data)
    % Recurse over node children.
    children = struct;
    if hasChildNodes(theNode)
        childNodes = getChildNodes(theNode);
        numChildNodes = getLength(childNodes);

        for count = 1:numChildNodes
            theChild = item(childNodes,count-1);
            [name,childs,Data] = getSegmentPointNodeData(theChild, Data);

            if (~strcmp(name,'#text') && ~strcmp(name,'#comment'))
                %add previously unknown (new) element to the structure

                if (isempty(fieldnames(childs)))
                    text = str2num(getTextContent(theChild));
                    if(~isempty(text))
                        children.(name) = text;
                    end
                else
                    children.(name) = childs;
                end
            end
        end
    end
end

% ----- Subfunction getSegmentPointNodeData -----
function [name,childs,Data] = getSegmentPointNodeData(theNode, Data)
    name = toCharArray(getNodeName(theNode))';
    if (~strcmp(name,'#text'))
        [attr] = parseSegmentAttributes(theNode);
        if (isempty(fieldnames(attr)))
            attr = [];
        end
        [childs,Data] = parseSegmentChildNodesPoints(theNode, Data);
        if (isfield(attr, 'label') && isfield(childs, 'pos_b'))
            Data.('points').(attr.label) = childs.pos_b;
        end
    else
        childs = struct;
    end
end

% ----- Subfunction getSegmentNodeData -----
function [name,childs,Data] = getSegmentNodeData(theNode, Data, numChildNodes, parentName, wb)
    name = toCharArray(getNodeName(theNode))';
    if (~strcmp(name,'#text'))
        [attr] = parseSegmentAttributes(theNode);

        if (strcmp(name,'frame'))
            [childs,Data] = parseSegmentChildNodes(theNode, Data, wb);
            if isfield(attr, 'index')
                frame = str2double(attr.index) + 1;
            else
                frame = NaN;
            end
            fields = fieldnames(childs);
            attributes = fieldnames(attr);
            if (isnan(frame))
                type = attr.type;
                type = strrep(type, '-', '_dash_');
                dataName =  'segmentData';
                for count2 = 1 : numel(fieldnames(childs))
                    fieldName = char(fields(count2));
                    values = sscanf(childs.(fieldName), '%f');
                    j = 0;
                    i = numel(values)/numel(Data.(dataName));
                    for count = 1 : numel(Data.(dataName))
                        Data.(dataName)(count).(type).(fields{count2, 1})(1, 1:i) = values(j+1:(j+i));
                        j =+ 1;
                    end
                end
            else
                if mod(frame, 200) == 0
                    val = frame / numChildNodes;
                    waitbar(val, wb);
                end

                for a = 2 : numel(fieldnames(attr))
                    Data.frame(frame).(char(attributes(a))) = attr.(char(attributes(a)));
                end

                for count2 = 1 : numel(fieldnames(childs))
                    fieldName = char(fields(count2));
                    values = sscanf(childs.(fieldName), '%f');
                    if contains(fieldName, 'sensor')
                        dataName = 'sensorData';
                    elseif contains(fieldName, 'Ergo')
                        dataName = 'ergonomicJointAngle';
                    elseif contains(fieldName, 'joint')
                        if contains(fieldName, 'Finger')
                            if contains(fieldName, 'Left')
                               side = 'Left';
                            else
                               side = 'Right';
                            end
                            dataName = ['fingerJointData' (side)];
                        else
                            dataName = 'jointData';
                        end
                    elseif contains(fieldName, 'foot')
                        dataName =  'footContact';
                    elseif contains(fieldName, 'Finger')
                        if contains(fieldName, 'Left')
                           side = 'Left';
                        else
                           side = 'Right';
                        end
                        dataName = ['fingerData' (side)];
                    elseif contains(fieldName, 'marker')
                        Data.frame(frame).('marker') = childs.(fieldName);
                        if ~isfield(Data, 'markers')
                            markerIdx = 1
                            Data.markers(markerIdx) = struct
                        else
                            markerIdx = 1 + length(Data.markers)
                        end
                        Data.markers(markerIdx).frame = frame
                        Data.markers(markerIdx).text = childs.(fieldName)
                        continue;
                    else
                        dataName = 'segmentData';
                    end
                    if ~isfield(Data, dataName)
%                         disp(dataName)
                        break;
%                         print('Unknown field: ' + dataName)
%                         print('Unknown field');
                    end

                    i = numel(values)/numel(Data.(dataName));
                    j = 0;
                    if (floor(i) == i)
                        if (frame > 1)
                            for count = 1 : numel(Data.(dataName))
                                Data.(dataName)(count).(fields{count2, 1})(frame, 1:i) = values(j+1:(j+i));
                                j = j + i;
                            end
                        elseif (frame == 1)
                            for count = 1 : numel(Data.(dataName))
                                totalFrames = floor(numChildNodes/2) - 3; %-3 for the identity, tpose and tpose-isb poses
                                Data.(dataName)(count).(fields{count2, 1}) = zeros(totalFrames, i);
                                Data.(dataName)(count).(fields{count2, 1})(frame, 1:i) = values(j+1:(j+i));
                                j = j + i;
                            end
                        end
                    else
                        Data.frame(frame).(fields{count2, 1}) = values;
                    end
                end
            end
        else
            if (isempty(fieldnames(attr)))
                %parse child nodes
                [childs,Data] = parseSegmentChildNodes(theNode, Data, wb);
            elseif (strcmp(name,'segment') || strcmp(name,'fingerTrackingSegment'))
                if contains(parentName, 'fingerTrackingSegments')
                    if contains(attr.label, 'Left')
                       side = 'Left';
                    else
                       side = 'Right';
                    end
                    fingerData = ['fingerData' (side)];
                    Data.(fingerData)(str2double(attr.index) + 1).('label') = attr.label;
                    [childs,segmentData2] = parseSegmentChildNodesPoints(theNode, Data.(fingerData)(str2double(attr.index) + 1));
                    Data.(fingerData)(str2double(attr.index) + 1).('points') = segmentData2.points;
                else
                    if ~isfield(attr, 'id')
                        print('oh nooooo')
                    end
                    Data.segmentData(str2double(attr.id)).('label') = attr.label;
                    [childs,segmentData2] = parseSegmentChildNodesPoints(theNode, Data.segmentData(str2double(attr.id)));
                    Data.segmentData(str2double(attr.id)).('points') = segmentData2.points;
                end
            elseif (strcmp(name,'sensor'))
                childs = struct;
                if ~isfield(Data, 'sensorData')
                    index = 1;
                else
                    index = numel(Data.sensorData) + 1;
                end
                Data.sensorData(index).('label') = attr.label;
            elseif (strcmp(name,'joint'))
                [childs,Data] = parseSegmentChildNodes(theNode, Data, wb);
                if contains(parentName, 'fingerTrackingJoints')
                    if contains(attr.label, 'Left')
                       side = 'Left';
                    else
                       side = 'Right';
                    end
                    fingerData = ['fingerJointData' (side)];
                    if ~isfield(Data, fingerData)
                        index = 1;
                    else
                        index = numel(Data.(fingerData)) + 1;
                    end
                    Data.(fingerData)(index).('label') = attr.label;
                    fields = fieldnames(childs);
                    for count2 = 1 : numel(fieldnames(childs))
                        Data.(fingerData)(index).(char(fields(count2))) = childs.(char(fields(count2)));
                    end
                else
                    if ~isfield(Data, 'jointData')
                        index = 1;
                    else
                        index = numel(Data.jointData) + 1;
                    end
                    Data.jointData(index).('label') = attr.label;
                    fields = fieldnames(childs);
                    for count2 = 1 : numel(fieldnames(childs))
                        Data.jointData(index).(char(fields(count2))) = childs.(char(fields(count2)));
                    end
                end
            elseif (strcmp(name,'ergonomicJointAngle'))
                childs = struct;
                if ~isfield(Data, 'ergonomicJointAngle')
                    index = 1;
                else
                    index = numel(Data.ergonomicJointAngle) + 1;
                end
                fields = fieldnames(attr);
                for count2 = 1 : numel(fieldnames(attr))
                    Data.ergonomicJointAngle(index).(char(fields(count2))) = attr.(char(fields(count2)));
                end
            elseif (strcmp(name,'contactDefinition'))
                childs = struct;
                Data.footContact(str2double(attr.index) + 1).('label') = attr.label;
            else
                %parse child nodes
                [childs,Data] = parseSegmentChildNodes(theNode, Data, wb);
            end
        end
    else
        childs = struct;
    end
end

% ----- Subfunction getNodeData -----
function [name, Data] = getNodeData(theNode, Data, wb)
    % Create structure of node info.

    %make sure name is allowed as structure name
    name = toCharArray(getNodeName(theNode))';
    name = strrep(name, '-', '_dash_');
    name = strrep(name, ':', '_colon_');
    name = strrep(name, '.', '_dot_');

    if (strcmp(name,'subject'))
        [Data] = parseMetaAttributes(theNode, Data, name);
        [~,Data] = parseSegmentChildNodes(theNode, Data, wb);
    else
        [Data] = parseMetaAttributes(theNode, Data, name);
        [Data] = parseChildNodes(theNode, Data, wb);
    end
end

%parse metadata attr and segment attr
% ----- Subfunction parseAttributes -----
function [Data] = parseMetaAttributes(theNode, Data, name)
    if hasAttributes(theNode)
       theAttributes = getAttributes(theNode);
       numAttributes = getLength(theAttributes);

       for count = 1:numAttributes
            str = toCharArray(toString(item(theAttributes,count-1)))';
            k = strfind(str,'=');
            attr_name = str(1:(k(1)-1));
            attr_name = strrep(attr_name, '-', '_dash_');
            attr_name = strrep(attr_name, ':', '_colon_');
            attr_name = strrep(attr_name, '.', '_dot_');
            total = strjoin({name, attr_name}, '_');
            Data.metaData.(char(total)) = str((k(1)+2):(end-1));
       end
    end
end

%parse metadata attr and segment attr
% ----- Subfunction parseSegmentAttributes -----
function [attributes] = parseSegmentAttributes(theNode)
    % Create attributes structure.

    if hasAttributes(theNode)
       theAttributes = getAttributes(theNode);
       numAttributes = getLength(theAttributes);

       for count = 1:numAttributes
            str = toCharArray(toString(item(theAttributes,count-1)))';
            k = strfind(str,'=');
            attr_name = str(1:(k(1)-1));
            attributes.(attr_name) = str((k(1)+2):(end-1));
       end
    else
       attributes = struct;
    end
end