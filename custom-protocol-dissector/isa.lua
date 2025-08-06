-- File:        isa.lua
-- Author:      Jan Matufka
-- Login:       xmatuf00
-- School:      FIT BUT
-- Description: Implementation of simple lua script for parsing Custom protocol packets.
-- Purpose:     ISA course assignment (Network applications and network administration) 

isa_protocol = Proto("ISAP", "ISA Protocol")   

payload     = ProtoField.string("isap.payload",     "Payload",              base.ASCII)
command     = ProtoField.string("isap.command",     "Used Command",         base.ASCII)
username    = ProtoField.string("isap.username",    "Username/Sender",      base.ASCII)
user_id     = ProtoField.string("isap.user_id",     "Caller ID (Base64)",   base.ASCII)
recipient   = ProtoField.string("isap.recipient",   "Recipient/Target",     base.ASCII)
password    = ProtoField.string("isap.password",    "Encrypted password",   base.ASCII)
subject     = ProtoField.string("isap.subject",     "Subject",              base.ASCII)
message     = ProtoField.string("isap.message",     "Message text",         base.ASCII)
fetch_id    = ProtoField.string("isap.fetch_id",    "Fetch ID",             base.ASCII)
return_code = ProtoField.string("isap.return_code", "Return code",          base.ASCII)
error_msg   = ProtoField.string("isap.error_msg",   "Info Message",         base.ASCII)

isa_protocol.fields = { payload, command, username, user_id, recipient, password, subject, message, fetch_id, return_code, error_msg }

response_table = {
    ["register"] = {command, username, password},
    ["login"] = {command, username, password},
    ["list"] = {command, user_id},
    ["send"] = {command, user_id, recipient, subject, message},
    ["fetch"] = {command, user_id, fetch_id},
    ["logout"] = {command, user_id},
    ["err"] = {return_code, error_msg},
}
-- ok responses from the server can have different number of data strings
-- and thus have to be parsed accordingly
ok_table = {
    [1] = {return_code},
    [2] = {return_code, error_msg}, -- 'register' or 'logout' or 'send' response
    [3] = {return_code, error_msg, user_id}, -- 'login' response
    [4] = {return_code, username, subject, message}, -- 'fetch' response
}
-- 'list' response contains other trees and is parsed differently so it wont be here
message_listing = {
    [1] = fetch_id,
    [2] = username,
    [3] = subject,
}

function isa_protocol.dissector(buffer, pinfo, tree)
    length = buffer:len()
    if length == 0 then 
        return 
    end
    -- convert wireshark buffer into lua string for easier manipulation
    data = buffer():string()
    pinfo.cols.protocol = isa_protocol.name
    local maintree = tree:add(isa_protocol, buffer(), "ISA Protocol Data")
    maintree:add(payload, buffer(0, length))
    local result = fsm(data)
    -- table data is an array containing data field strings and in case of 
    -- list response can also contain another arrays
    local table_data = result[1]
    -- idx_data lists indices of each parsed word => starting index, and index at the end of the word
    -- so if the buffer contains 3 important data fields, the idx_data would look like this:
    -- {1_start, 1_end, 2_start, 2_end, 3_start, 3_end,}
    local idx_data = result[2]

    local selector = table_data[1]

    local idx = 1 -- for indexing in the idx_data
    local message_id = 1 -- for helping indexing inside nested message info fields

    -- ok response parsing can have different structures based on which command it is responding to
    if selector == "ok" then
        local length = #table_data -- for helping decide ok response data fields
        for i, data in pairs(table_data) do

            -- start_byte = needed for setting the buffer() parameters to show in Wireshark
            local start_byte = idx_data[2*idx-1]
            
            -- strings are parsed differently than nested message info arrays
            local typeof = type(data)

            -- normal field
            if typeof == "string" then
                local something = maintree:add(ok_table[length][i], buffer(start_byte, data:len()))
            else 
                -- idx_data = start index and end index of each word, list message starts with 'ok'
                -- one message contains 3 entries => each have 2 indices, hence multiplying by 6
                local msg_start = idx_data[6*message_id-3]
                local msg_end = idx_data[6*message_id+2]
                local msg_len = msg_end - msg_start + 3 -- to show it with starting ( and ending ")
                local message = maintree:add(isa_protocol, buffer(msg_start-1, msg_len), "Message "..message_id.." data")
                for j, subtree_data in pairs(data) do
                    -- idx_data = ok1 ok2 (id1 id2 us1 us2 sub1 sub2) (id1 id2 us1 us2 sub1 sub2) ...
                    -- for message_id = 1 we want starting indices at 3,5,7 for j = 1,2,3
                    -- for message_id = 2 we want starting indices at 9,11,13 for j = 1,2,3 ...
                    start_byte = idx_data[6*message_id+2*j - 5]
                    message:add(message_listing[j], buffer(start_byte, subtree_data:len()))
                    idx = idx + 1
                end
                message_id = message_id + 1
            end
            idx = idx + 1
        end
    else 
        -- other than ok responses (just basic parsing)
        for i, data in pairs(table_data) do
            local start_byte = idx_data[2*idx-1]
            maintree:add(response_table[selector][i], buffer(start_byte, data:len()))
            idx = idx + 1
        end
    end
end

-- returns an array of important data arrays (data strings, indexes and subtrees)
function fsm(data)
    -- highest level of data => contains data strings and in some cases, 
    local result_table = {}
    
    -- starting indices for each data item (used later for parsing)
    local index_table = {}

    -- one item from response to 'list' command
    local subtree = {}

    -- initializing state machine data
    local current_word = ""
    local state = "s_start"
    local next_state = "s_start"
    local start_idx = 0
    local end_idx = 0
    for c in data:gmatch(".") do
        state = next_state
        -- initial state
        if state == "s_start" then
            if c == ' ' then next_state = "s_start" start_idx = start_idx + 1
            elseif c == '(' then next_state = "s_word" start_idx = start_idx + 1
            end
        -- word state
        elseif state == "s_word" then
            if c == '(' then
                start_idx = start_idx + 1
                next_state = "s_list"
            elseif c == ')' then 
                if current_word:len() > 0 then
                    table.insert(result_table, current_word)
                    table.insert(index_table, start_idx)
                    table.insert(index_table, end_idx)
                    start_idx = end_idx + 1
                    current_word = ""
                end
                next_state = "s_end"
            elseif c == '"' then 
                start_idx = start_idx + 1
                next_state = "s_str1"
            elseif c == ' ' then
                if current_word:len() > 0 then
                    table.insert(result_table, current_word)
                    table.insert(index_table, start_idx)
                    table.insert(index_table, end_idx)
                    start_idx = end_idx + 1
                    current_word = ""
                else 
                    start_idx = start_idx + 1
                end
            else current_word = current_word .. c 
            end
        -- string state (inside "...") - within basic command
        elseif state == "s_str1" then
            if c == '"' then
                table.insert(result_table, current_word)
                table.insert(index_table, start_idx)
                table.insert(index_table, end_idx)
                start_idx = end_idx + 1
                current_word = ""
                next_state  = "s_word"
            elseif c == '\\' then next_state = "s_esc1"
            else current_word = current_word .. c 
            end
        -- escaping character inside str1
        elseif state == "s_esc1" then
            if c == '"' then current_word = current_word .. c
            elseif c == 'n' then current_word = current_word .. '\n'
            elseif c == 't' then current_word = current_word .. '\t'
            elseif c == '\\' then current_word = current_word .. '\\'
            else return nil
            end
            next_state  = "s_str1"
        -- list command state
        elseif state == "s_list" then
            if c == '(' then 
                -- TODO: create subtree
                start_idx = start_idx + 1
                next_state = "s_list_item"
            elseif c == ')' then 
                -- TODO: save list of subtrees
                start_idx = start_idx + 1
                next_state = "s_word" 
            elseif c == '"' then
                start_idx = start_idx + 1
                next_state = "s_str2" 
            elseif c == ' ' then
                start_idx = start_idx + 1
            end
        -- fetch response has one inner parenthesis
        elseif state == "s_str2" then 
            if c == '"' then
                table.insert(result_table, current_word)
                table.insert(index_table, start_idx)
                table.insert(index_table, end_idx)
                start_idx = end_idx + 1
                current_word = ""
                next_state  = "s_list"
            elseif c == '\\' then next_state = "s_esc2"
            else current_word = current_word .. c 
            end
        -- escaping character inside str2
        elseif state == "s_esc2" then
            if c == '"' then current_word = current_word .. c
            elseif c == 'n' then current_word = current_word .. '\n'
            elseif c == 't' then current_word = current_word .. '\t'
            elseif c == '\\' then current_word = current_word .. '\\'
            else return nil
            end
            next_state = "s_str2"
        -- list entry state
        elseif state == "s_list_item" then
            if c == ')' then 
                table.insert(result_table, subtree)
                subtree = {}
                start_idx = start_idx + 1
                next_state = "s_list"
            elseif c == ' ' then
                if current_word:len() > 0 then
                    table.insert(subtree, current_word)
                    table.insert(index_table, start_idx)
                    table.insert(index_table, end_idx)
                    start_idx = end_idx + 1
                    current_word = ""
                else 
                    start_idx = start_idx + 1
                end
            elseif c == '"' then 
                start_idx = start_idx + 1
                next_state = "s_str3"
            else current_word = current_word .. c 
            end
        -- string state (inside "...") - within list entry
        elseif state == "s_str3" then 
            if c == '"' then
                table.insert(subtree, current_word)
                table.insert(index_table, start_idx)
                table.insert(index_table, end_idx)
                start_idx = end_idx + 1
                current_word = ""
                next_state  = "s_list_item"
            elseif c == '\\' then next_state = "s_esc3"
            else current_word = current_word .. c 
            end
        -- escaping character inside str2
        elseif state == "s_esc3" then
            if c == '"' then current_word = current_word .. c
            elseif c == 'n' then current_word = current_word .. '\n'
            elseif c == 't' then current_word = current_word .. '\t'
            elseif c == '\\' then current_word = current_word .. '\\'
            else return nil
            end
            next_state = "s_str3"
        -- last parenthesis ')' - ending a command and the cycle
        elseif state == "s_end" then
            -- table.insert(result_table, current_word)
            -- table.insert(index_table, start_idx)
            -- table.insert(index_table, end_idx)
            -- return {result_table, index_table}
        else
        end
        end_idx = end_idx + 1
    end
    return {result_table, index_table}
end

local tcp_port = DissectorTable.get("tcp.port")
tcp_port:add(32323, isa_protocol)
